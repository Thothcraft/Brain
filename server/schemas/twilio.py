"""Pydantic models for Twilio integration."""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class MessageStatus(str, Enum):
    """Status of a Twilio message."""
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    UNDELIVERED = "undelivered"
    FAILED = "failed"
    RECEIVING = "receiving"
    RECEIVED = "received"
    ACCEPTED = "accepted"
    SCHEDULED = "scheduled"
    READ = "read"
    CANCELED = "canceled"


class MessageDirection(str, Enum):
    """Direction of a Twilio message."""
    INBOUND = "inbound"
    OUTBOUND_API = "outbound-api"
    OUTBOUND_CALL = "outbound-call"
    OUTBOUND_REPLY = "outbound-reply"


class MessageType(str, Enum):
    """Type of message content."""
    TEXT = "text"
    MEDIA = "media"
    VOICE = "voice"


class MessageRequest(BaseModel):
    """Request model for sending a message via Twilio."""
    to: str = Field(..., example="+1234567890", 
                   description="Recipient's phone number in E.164 format")
    body: str = Field(..., example="Hello from the app!", 
                     description="Message content")
    from_: Optional[str] = Field(None, alias="from", 
                               example="+1987654321",
                               description="Sender's phone number (defaults to TWILIO_PHONE_NUMBER)")
    media_url: Optional[List[HttpUrl]] = Field(None,
                                             description="List of media URLs to send with the message")
    status_callback: Optional[HttpUrl] = Field(None,
                                             description="Webhook URL for status updates")
    max_price: Optional[float] = Field(None, ge=0.0, le=100.0,
                                     description="Maximum price in USD to pay for the message")
    validity_period: Optional[int] = Field(14400, ge=1, le=14400,
                                          description="How long the message is valid in seconds")
    
    @validator('to', 'from_')
    def validate_phone_number(cls, v):
        if v and not v.startswith('+'):
            raise ValueError('Phone numbers must be in E.164 format (e.g., +1234567890)')
        return v


class MessageResponse(BaseModel):
    """Response model for a sent message."""
    sid: str = Field(..., example="SMXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
                    description="Unique identifier for the message")
    status: MessageStatus = Field(..., 
                                example=MessageStatus.QUEUED,
                                description="Current status of the message")
    date_created: datetime = Field(..., 
                                 example="2023-01-01T12:00:00Z",
                                 description="When the message was created")
    date_sent: Optional[datetime] = Field(None,
                                        example="2023-01-01T12:00:05Z",
                                        description="When the message was sent")
    date_updated: datetime = Field(...,
                                 example="2023-01-01T12:00:00Z",
                                 description="When the message was last updated")
    direction: MessageDirection = Field(...,
                                      example=MessageDirection.OUTBOUND_API,
                                      description="Direction of the message")
    from_: str = Field(..., alias="from", 
                      example="+1987654321",
                      description="Sender's phone number")
    to: str = Field(..., example="+1234567890",
                   description="Recipient's phone number")
    body: Optional[str] = Field(None, 
                               example="Hello from the app!",
                               description="Message content")
    num_media: int = Field(0, example=0,
                          description="Number of media items in the message")
    num_segments: int = Field(1, example=1,
                             description="Number of segments the message was split into")
    price: Optional[float] = Field(None, example=0.0075,
                                  description="Price of the message in USD")
    error_code: Optional[int] = Field(None, example=12345,
                                     description="Error code if message failed")
    error_message: Optional[str] = Field(None,
                                        example="Unknown error",
                                        description="Error description")
    uri: str = Field(..., 
                    example="/2010-04-01/Accounts/ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/Messages/SMXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.json",
                    description="URI of the message resource")


class IncomingMessage(BaseModel):
    """Model for incoming Twilio SMS/webhook messages."""
    message_sid: str = Field(..., example="SMXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    account_sid: str = Field(..., example="ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    from_: str = Field(..., alias="from", example="+1234567890")
    to: str = Field(..., example="+1987654321")
    body: str = Field(..., example="Hello!")
    num_media: str = Field("0", example="0")
    media_content_type: Optional[str] = Field(None, example="image/jpeg")
    media_url: Optional[HttpUrl] = Field(None)
    message_status: Optional[MessageStatus] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    date_created: Optional[datetime] = None
    date_sent: Optional[datetime] = None
    date_updated: Optional[datetime] = None
    direction: Optional[MessageDirection] = None
    price: Optional[float] = None
    price_unit: Optional[str] = Field(None, example="USD")
    api_version: Optional[str] = Field(None, example="2010-04-01")


class CallStatus(str, Enum):
    """Status of a Twilio call."""
    QUEUED = "queued"
    RINGING = "ringing"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    BUSY = "busy"
    FAILED = "failed"
    NO_ANSWER = "no-answer"
    CANCELED = "canceled"


class CallRequest(BaseModel):
    """Request model for making a call via Twilio."""
    to: str = Field(..., example="+1234567890",
                   description="The phone number to call")
    from_: Optional[str] = Field(None, alias="from",
                               example="+1987654321",
                               description="Caller ID (defaults to TWILIO_PHONE_NUMBER)")
    url: HttpUrl = Field(...,
                        example="https://example.com/twilio/voice/answer",
                        description="URL that returns TwiML instructions for the call")
    method: str = Field("POST",
                       description="HTTP method to use when requesting the URL")
    status_callback: Optional[HttpUrl] = Field(None,
                                             description="Webhook URL for call status changes")
    status_callback_event: Optional[List[str]] = Field(None,
                                                     description="Which status changes to receive callbacks for")
    status_callback_method: Optional[str] = Field(None,
                                                description="HTTP method to use for status callbacks")
    timeout: Optional[int] = Field(30, ge=5, le=600,
                                  description="Seconds to wait for answer")
    record: bool = Field(False, description="Whether to record the call")
    recording_status_callback: Optional[HttpUrl] = Field(None,
                                                       description="Webhook URL for recording status changes")
    recording_status_callback_method: Optional[str] = Field(None,
                                                          description="HTTP method for recording callbacks")
    machine_detection: Optional[str] = Field(None,
                                           description="Enable machine detection")
    machine_detection_timeout: Optional[int] = Field(30, ge=3, le=30,
                                                   description="Seconds to wait for machine detection")
    
    @validator('to', 'from_')
    def validate_phone_number(cls, v):
        if v and not v.startswith('+'):
            raise ValueError('Phone numbers must be in E.164 format (e.g., +1234567890)')
        return v


class CallResponse(BaseModel):
    """Response model for a call."""
    sid: str = Field(..., example="CAXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    status: CallStatus = Field(..., example=CallStatus.QUEUED)
    from_: str = Field(..., alias="from", example="+1987654321")
    to: str = Field(..., example="+1234567890")
    date_created: datetime = Field(..., example="2023-01-01T12:00:00Z")
    date_updated: datetime = Field(..., example="2023-01-01T12:00:00Z")
    start_time: Optional[datetime] = Field(None, example="2023-01-01T12:00:05Z")
    end_time: Optional[datetime] = Field(None, example="2023-01-01T12:02:30Z")
    duration: Optional[int] = Field(None, example=145, description="Call duration in seconds")
    price: Optional[float] = Field(None, example=0.0100, description="Call cost in USD")
    price_unit: Optional[str] = Field(None, example="USD")
    direction: str = Field(..., example="outbound-api")
    answered_by: Optional[str] = Field(None, example="human")
    forwarded_from: Optional[str] = Field(None)
    caller_name: Optional[str] = Field(None)
    uri: str = Field(..., example="/2010-04-01/Accounts/ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/Calls/CAXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.json")


class TranscriptionStatus(str, Enum):
    """Status of a transcription."""
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionResponse(BaseModel):
    """Response model for a transcription."""
    sid: str = Field(..., example="TRXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    status: TranscriptionStatus = Field(..., example=TranscriptionStatus.COMPLETED)
    transcription_text: str = Field(..., example="Hello, this is a test transcription.")
    recording_sid: Optional[str] = Field(None, example="REXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    call_sid: Optional[str] = Field(None, example="CAXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    date_created: datetime = Field(..., example="2023-01-01T12:00:00Z")
    date_updated: datetime = Field(..., example="2023-01-01T12:00:00Z")
    duration: Optional[int] = Field(None, example=5, description="Transcription duration in seconds")
    price: Optional[float] = Field(None, example=0.05, description="Transcription cost in USD")
    price_unit: Optional[str] = Field(None, example="USD")
    language: Optional[str] = Field(None, example="en-US")
    uri: str = Field(..., example="/2010-04-01/Accounts/ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/Transcriptions/TRXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.json")
