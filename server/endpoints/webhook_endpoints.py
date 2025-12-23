"""Webhook endpoints for external service integrations (Twilio, etc.)."""

import re
import json
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Form
from sqlalchemy.orm import Session
from twilio.twiml.voice_response import VoiceResponse

from server.db import get_db, User, Query as DBQuery
from server.utils.logging_utils import log_request_start, log_response, log_error, log_request_payload
from aiagent.handler.query import query_openai
from aiagent.memory.memory_manager import LongTermMemoryManager, ShortTermMemoryManager

router = APIRouter(prefix="/phone", tags=["webhooks"])

@router.post(
    "/incoming-message",
    summary="Handle Twilio incoming SMS",
    description="Process incoming SMS messages from Twilio webhook",
    responses={
        200: {"description": "Message processed successfully"},
        400: {"description": "Invalid message format"},
        401: {"description": "Unauthorized phone number"}
    }
)
async def handle_twilio_incoming_message(
    request: Request, 
    db: Session = Depends(get_db)
) -> Response:
    """
    Handle incoming SMS messages from Twilio and forward them to the query endpoint.
    
    Purpose: Process SMS messages and provide AI responses via Twilio
    
    Args:
        request: The incoming HTTP request containing Twilio webhook data
        db: Database session
        
    Returns:
        Response: TwiML response for Twilio
    """
    try:
        # Parse the form data from the request
        form_data = await request.form()
        from_number = form_data.get('From', '')
        body = form_data.get('Body', '').strip()
        
        endpoint_name = "/api/webhooks/twilio/incoming-message"
        log_request_start(endpoint_name, "POST", dict(request.headers), request.client.host if request.client else "unknown")
        log_request_payload(dict(form_data), endpoint_name)
        
        # Log the incoming message
        log_response(200, f"Incoming message from {from_number}: {body}", "/webhooks/twilio/incoming-message")
        
        # Normalize phone number (remove non-digits)
        normalized_from = re.sub(r'\D', '', from_number)
        
        # Find user by phone number
        user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from.isdigit() else None
        
        if not user:
            log_error(f"Unauthorized access attempt from {from_number}")
            return Response(
                content="<Response><Message>Unauthorized. Please register first.</Message></Response>",
                media_type="application/xml",
                status_code=200
            )
        
        # Create a chat ID for this conversation
        chat_id = f"sms_{normalized_from}"
        user_query_text = body
        
        # Save the query to database
        try:
            db_query = DBQuery(
                userId=user.userId,
                chatId=chat_id,
                query_text=user_query_text,
                response=None  # Will be updated with AI response
            )
            db.add(db_query)
            db.commit()
            db.refresh(db_query)
            
        except Exception as e:
            log_error(f"Database error: {str(e)}")
            return Response(
                content="<Response><Message>Service temporarily unavailable.</Message></Response>",
                media_type="application/xml",
                status_code=200
            )
        
        # Process with AI (simplified version)
        try:
            # Use basic memory managers for SMS
            long_term_memory = LongTermMemoryManager()
            short_term_memory = ShortTermMemoryManager()
            
            ai_response = query_openai(
                query=user_query_text,
                long_term_memory=long_term_memory,
                short_term_memory=short_term_memory,
                max_tokens=160,  # SMS character limit consideration
                temperature=0.7,
                aux_data={"current_user_id": user.userId, "source": "sms"}
            )
            
            # Update query with response
            db_query.response = ai_response
            db.commit()
            
            # Truncate response if too long for SMS
            if len(ai_response) > 1500:  # Leave room for TwiML overhead
                ai_response = ai_response[:1500] + "..."
            
        except Exception as e:
            log_error(f"AI processing error: {str(e)}")
            ai_response = "I'm sorry, I'm experiencing technical difficulties. Please try again later."
        
        # Return TwiML response
        twiml_response = f"<Response><Message>{ai_response}</Message></Response>"
        
        log_response(200, "SMS response sent", "/webhooks/twilio/incoming-message")
        return Response(
            content=twiml_response,
            media_type="application/xml",
            status_code=200
        )
        
    except Exception as e:
        log_error(f"Webhook error: {str(e)}")
        return Response(
            content="<Response><Message>Error processing message.</Message></Response>",
            media_type="application/xml",
            status_code=200
        )

@router.post(
    "/message-status",
    summary="Handle Twilio message status",
    description="Process message delivery status updates from Twilio",
    responses={
        200: {"description": "Status update processed"}
    }
)
async def handle_twilio_message_status(
    request: Request,
    MessageSid: str = Form(...),
    MessageStatus: str = Form(...),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Handle Twilio message status updates.
    
    Purpose: Track SMS delivery status for monitoring and debugging
    
    Args:
        request: HTTP request
        MessageSid: Twilio message ID
        MessageStatus: Message delivery status
        db: Database session
        
    Returns:
        Dict: Status acknowledgment
    """
    try:
        log_request_start("POST", "/api/webhooks/twilio/message-status", None)
        
        # Log the status update
        log_response(200, f"Message {MessageSid} status: {MessageStatus}", "/webhooks/twilio/message-status")
        
        # You could store this in a message_status table if needed
        # For now, just log it
        
        return {"status": "received"}
        
    except Exception as e:
        log_error(f"Message status webhook error: {str(e)}")
        return {"status": "error"}

@router.post(
    "/incoming-call",
    summary="Handle Twilio incoming calls",
    description="Process incoming voice calls from Twilio webhook",
    responses={
        200: {"description": "Call handled successfully"}
    }
)
async def handle_twilio_incoming_call(
    request: Request,
    From: str = Form(None),
    To: str = Form(None),
    db: Session = Depends(get_db)
) -> Response:
    """
    Handle incoming voice calls from Twilio.
    
    Purpose: Provide voice interaction capabilities via Twilio
    
    Args:
        request: HTTP request
        From: Caller's phone number
        To: Called number
        db: Database session
        
    Returns:
        Response: TwiML response for call handling
    """
    try:
        log_request_start("POST", "/webhooks/twilio/incoming-call", None)
        
        # Normalize phone number
        normalized_from = re.sub(r'\D', '', From) if From else ""
        
        # Find user by phone number
        user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from.isdigit() else None
        
        # Create TwiML response
        resp = VoiceResponse()
        
        if not user:
            resp.say("Sorry, this number is not registered. Please register first.")
            resp.hangup()
        else:
            resp.say(f"Hello {user.username}! Welcome to the AI assistant.")
            resp.say("Please speak your question after the beep, and I'll provide an answer.")
            
            # Record the user's question
            resp.record(
                timeout=10,
                transcribe=True,
                transcribe_callback="/webhooks/twilio/transcription-callback",
                max_length=30
            )
            
            resp.say("Thank you. Processing your request.")
        
        log_response(200, "Call handled", "/webhooks/twilio/incoming-call")
        return Response(content=str(resp), media_type="application/xml", status_code=200)
        
    except Exception as e:
        log_error(f"Call webhook error: {str(e)}")
        resp = VoiceResponse()
        resp.say("Sorry, there was an error processing your call.")
        resp.hangup()
        return Response(content=str(resp), media_type="application/xml", status_code=200)

@router.post(
    "/transcription-callback",
    summary="Handle Twilio transcription callback",
    description="Process voice transcription from Twilio and provide AI response",
    responses={
        200: {"description": "Transcription processed successfully"}
    }
)
async def handle_transcription_callback(
    request: Request,
    From: str = Form(...),
    TranscriptionText: str = Form(...),
    TranscriptionStatus: str = Form(...),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Handle voice transcription callback from Twilio.
    
    Purpose: Process voice-to-text transcription and provide AI responses
    
    Args:
        request: HTTP request
        From: Caller's phone number
        TranscriptionText: Transcribed text
        TranscriptionStatus: Transcription status
        db: Database session
        
    Returns:
        Dict: Processing acknowledgment
    """
    try:
        log_request_start("POST", "/webhooks/twilio/transcription-callback", None)
        
        if TranscriptionStatus != "completed":
            log_error(f"Transcription failed: {TranscriptionStatus}")
            return {"status": "transcription_failed"}
        
        # Normalize phone number
        normalized_from = re.sub(r'\D', '', From)
        
        # Find user by phone number
        user = db.query(User).filter(User.phone_number == int(normalized_from)).first() if normalized_from.isdigit() else None
        
        if not user:
            log_error(f"User not found for transcription callback: {From}")
            return {"status": "user_not_found"}
        
        # Process transcription with AI
        try:
            # Save query to database
            chat_id = f"voice_{normalized_from}"
            db_query = DBQuery(
                userId=user.userId,
                chatId=chat_id,
                query_text=TranscriptionText,
                response=None
            )
            db.add(db_query)
            db.commit()
            db.refresh(db_query)
            
            # Process with AI
            long_term_memory = LongTermMemoryManager()
            short_term_memory = ShortTermMemoryManager()
            
            ai_response = query_openai(
                query=TranscriptionText,
                long_term_memory=long_term_memory,
                short_term_memory=short_term_memory,
                max_tokens=500,
                temperature=0.7,
                aux_data={"current_user_id": user.userId, "source": "voice"}
            )
            
            # Update query with response
            db_query.response = ai_response
            db.commit()
            
            # Send SMS response (since voice callback is async)
            try:
                from server.services import send_twilio_message
                send_twilio_message(From, f"Voice Response: {ai_response}")
            except Exception as sms_error:
                log_error(f"Failed to send SMS response: {str(sms_error)}")
            
            log_response(200, "Transcription processed and response sent", "/webhooks/twilio/transcription")
            return {"status": "processed"}
            
        except Exception as e:
            log_error(f"AI processing error in transcription: {str(e)}")
            return {"status": "processing_error"}
        
    except Exception as e:
        log_error(f"Transcription callback error: {str(e)}")
        return {"status": "error"}
