"""Pydantic models for AI-related endpoints and functionality."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for AI queries."""
    query: str = Field(..., example="What is the capital of France?", 
                      description="The user's query to process")
    chat_id: Optional[str] = Field(None, example="chat_abc123", 
                                  description="Optional chat ID for conversation history")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, 
                                       description="Sampling temperature for response generation")
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000, 
                                     description="Maximum number of tokens to generate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "chat_id": "chat_abc123",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }


class QueryResponse(BaseModel):
    """Response model for AI queries."""
    response: str = Field(..., example="The capital of France is Paris.", 
                         description="The AI's response to the query")
    chat_id: str = Field(..., example="chat_abc123", 
                        description="The chat ID for this conversation")
    tokens_used: int = Field(..., example=42, 
                            description="Number of tokens used in the response")
    model: str = Field(..., example="gpt-3.5-turbo", 
                      description="The AI model used for generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "The capital of France is Paris.",
                "chat_id": "chat_abc123",
                "tokens_used": 42,
                "model": "gpt-3.5-turbo"
            }
        }


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str = Field(..., example="user", description="The role of the message sender (user/assistant)")
    content: str = Field(..., example="Hello, how are you?", description="The message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, 
                              description="When the message was sent")
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ["user", "assistant", "system"]:
            raise ValueError("Role must be one of 'user', 'assistant', or 'system'")
        return v


class ChatHistory(BaseModel):
    """A conversation history between user and AI."""
    chat_id: str = Field(..., example="chat_abc123", 
                        description="Unique identifier for this chat")
    user_id: int = Field(..., example=1, 
                        description="ID of the user who owns this chat")
    title: Optional[str] = Field(None, example="Discussion about France",
                                description="Optional title for the chat")
    messages: List[ChatMessage] = Field(default_factory=list,
                                      description="List of messages in the chat")
    created_at: datetime = Field(default_factory=datetime.utcnow,
                               description="When the chat was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow,
                               description="When the chat was last updated")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                    description="Additional metadata for the chat")


class AIConfig(BaseModel):
    """Configuration for AI model parameters."""
    model_name: str = Field("gpt-3.5-turbo", 
                           description="Name of the AI model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0,
                              description="Controls randomness in generation")
    max_tokens: int = Field(1000, ge=1, 
                           description="Maximum number of tokens to generate")
    top_p: float = Field(1.0, ge=0.0, le=1.0,
                        description="Nucleus sampling parameter")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0,
                                    description="Penalize new tokens based on frequency")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0,
                                   description="Penalize new tokens based on presence")
    stop: Optional[List[str]] = Field(None, 
                                     description="Sequences where the API will stop generating")


class AITrainingExample(BaseModel):
    """A single training example for fine-tuning the AI model."""
    prompt: str = Field(..., description="The input prompt")
    completion: str = Field(..., description="The desired completion")
    metadata: Optional[Dict[str, Any]] = Field(None, 
                                             description="Additional metadata for the example")


class AITrainingRequest(BaseModel):
    """Request model for submitting training data to fine-tune the AI."""
    examples: List[AITrainingExample] = Field(..., 
                                            description="List of training examples")
    model_name: Optional[str] = Field("gpt-3.5-turbo",
                                     description="Base model to fine-tune")
    validation_split: float = Field(0.1, ge=0.0, le=0.5,
                                   description="Fraction of data to use for validation")
    epochs: int = Field(3, ge=1, description="Number of training epochs")
    batch_size: Optional[int] = Field(None, 
                                     description="Batch size for training")
    learning_rate: float = Field(1e-5, ge=0, 
                                description="Learning rate for training")
