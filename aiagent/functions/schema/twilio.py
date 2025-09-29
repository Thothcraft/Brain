from typing import Dict, Any
from pydantic import BaseModel, Field
import json
from datetime import datetime

# Use the shared function schema decorator (compat shim re-exports correct impl)
from server.utils.functions_metadata import function_schema

# We delegate to the backend service that already wraps Twilio
from server.services import send_twilio_message as _send_twilio_message
from server.db import SessionLocal, File as DBFile


class TwilioSendMessageResponse(BaseModel):
    message_sid: str = Field(..., description="Twilio message SID")
    status: str | None = Field(None, description="Delivery status returned by Twilio")
    to: str | None = Field(None, description="Recipient phone number")
    date_created: str | None = Field(None, description="Creation timestamp from Twilio")


@function_schema(
    name="send_twilio_message",
    description=(
        "Send an SMS via Twilio to a target phone number. "
        "Use E.164 format for the phone number (e.g., +14155552671)."
    ),
    required_params=["to_phone_number", "message"],
)
def send_twilio_message(to_phone_number: str, message: str) -> Dict[str, Any]:
    """
    AI tool: Send an SMS using the server's Twilio integration.

    Args:
        to_phone_number: Recipient phone number in E.164 format (e.g., +14155552671)
        message: Text message content

    Returns:
        A dictionary with Twilio message metadata (SID, status, etc.)
    """
    # Delegate to the server service which handles auth, errors, and config
    result = _send_twilio_message(to_phone_number=to_phone_number, message=message)

    # Ensure the response is JSON-serialisable and matches our model shape
    payload = {
        "message_sid": result.get("message_sid"),
        "status": result.get("status"),
        "to": result.get("to"),
        "date_created": result.get("date_created"),
    }

    # Best-effort: record the sent SMS into the current user's short-term memory
    try:
        # CURRENT_USER_ID is set by aiagent.handler.query.query_openai via globals()
        user_id = globals().get("CURRENT_USER_ID")
        if user_id is not None:
            db = SessionLocal()
            try:
                stm_file = db.query(DBFile).filter(DBFile.userId == user_id, DBFile.filename == "short_term_memory.json").first()
                memory = {}
                if stm_file and stm_file.content:
                    try:
                        memory = json.loads(stm_file.content.decode("utf-8"))
                    except Exception:
                        memory = {}

                # Update conversations with a synthetic tool exchange
                conversations = memory.get("conversations", [])
                tool_query = f"TOOL CALL: send_twilio_message(to={to_phone_number})"
                tool_response = f"SMS sent with SID {payload.get('message_sid')} to {payload.get('to')}"
                conversations.append({
                    "query": tool_query,
                    "response": tool_response,
                    "summary": "Sent SMS via Twilio"
                })
                if len(conversations) > 50:
                    conversations = conversations[-50:]
                memory["conversations"] = conversations

                # Maintain a sent_sms list to help the agent avoid repeats
                sent_sms = memory.get("sent_sms", [])
                sent_sms.append({
                    "to": payload.get("to") or to_phone_number,
                    "message": message,
                    "sid": payload.get("message_sid"),
                    "date": payload.get("date_created") or datetime.utcnow().isoformat(),
                })
                if len(sent_sms) > 100:
                    sent_sms = sent_sms[-100:]
                memory["sent_sms"] = sent_sms

                # Persist
                encoded = json.dumps(memory).encode("utf-8")
                if stm_file:
                    stm_file.content = encoded
                    stm_file.size = len(encoded)
                else:
                    new_file = DBFile(
                        userId=user_id,
                        filename="short_term_memory.json",
                        content=encoded,
                        content_type="application/json",
                    )
                    new_file.size = len(encoded)
                    db.add(new_file)
                db.commit()
            finally:
                db.close()
    except Exception:
        # Do not fail tool on memory update issues
        pass

    return payload
