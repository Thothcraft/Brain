"""AI query processing endpoints."""

import json
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, validator

from server.db import get_db, User, File, Query as DBQuery
from server.auth import get_current_user
from server.utils.logging_utils import log_request_start, log_response, log_error, log_ai_call, log_ai_response
from aiagent.handler.query import query_openai
from aiagent.memory.memory_manager import LongTermMemoryManager, ShortTermMemoryManager

router = APIRouter(prefix="", tags=["ai"])

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) < 1:
            raise ValueError('Query cannot be empty')
        if len(v) > 10000:  # Prevent extremely long queries
            raise ValueError('Query too long (max 10,000 characters)')
        return v.strip()

class QueryResponse(BaseModel):
    success: bool
    query_id: int
    response: str
    chat_id: str
    timestamp: str

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process AI query",
    description="Process a query using AI and return the response",
    responses={
        200: {"description": "Query processed successfully"},
        400: {"description": "Invalid query"},
        401: {"description": "Not authenticated"},
        500: {"description": "AI processing error"}
    }
)
async def process_ai_query(
    request: Request,
    query_data: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Process an AI query and return the response.
    
    Purpose: Handle AI query processing with memory management and conversation tracking
    
    Args:
        request: HTTP request object
        query_data: Query text and optional context
        current_user: Authenticated user
        db: Database session
        
    Returns:
        QueryResponse: AI response and query metadata
    """
    try:
        log_request_start("POST", "/query", current_user.userId)
        log_ai_call(query_data.query, "default_model", "/query")
        
        # Generate chat_id if not provided
        chat_id = query_data.chat_id or f"chat_{current_user.userId}_{int(datetime.now().timestamp())}"
        
        # Save query to database
        db_query = DBQuery(
            userId=current_user.userId,
            chatId=chat_id,
            query_text=query_data.query,
            response=None  # Will be updated with AI response
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)
        
        # Get or create memory files
        def get_or_create_memory_file(filename, default_content='{}'):
            file = db.query(File).filter(
                File.userId == current_user.userId, 
                File.filename == filename
            ).first()
            if not file:
                file = File(
                    userId=current_user.userId,
                    filename=filename,
                    content=default_content.encode('utf-8'),
                    content_type='application/json',
                    size=len(default_content),
                    uploaded_at=datetime.now()
                )
                db.add(file)
                db.commit()
                db.refresh(file)
            return file
        
        # Initialize simple memory storage (without external memory managers)
        try:
            longterm_file = get_or_create_memory_file("long_term_memory.json")
            shortterm_file = get_or_create_memory_file("short_term_memory.json")
            
            lt_str = longterm_file.content.decode('utf-8') if longterm_file.content else "{}"
            st_str = shortterm_file.content.decode('utf-8') if shortterm_file.content else "{}"
            
            lt_data = json.loads(lt_str) if lt_str.strip() else {}
            st_data = json.loads(st_str) if st_str.strip() else {}
            
            # Inject user's phone number from database into long-term memory context
            if current_user.phone_number:
                lt_data["user_phone_number"] = f"+{current_user.phone_number}"
            lt_data["username"] = current_user.username
            lt_data["user_id"] = current_user.userId
            
        except Exception as e:
            log_error(f"Memory initialization error: {str(e)}")
            # Fallback to empty memory
            lt_data = {}
            st_data = {}
        
        # Process query with AI
        try:
            # Create memory managers with the loaded data
            long_term_memory = LongTermMemoryManager()
            long_term_memory._memory_content = lt_data
            short_term_memory = ShortTermMemoryManager()
            short_term_memory._memory_content = st_data
            
            ai_response = query_openai(
                query=query_data.query,
                long_term_memory=long_term_memory,
                short_term_memory=short_term_memory,
                max_tokens=2000,
                temperature=0.7,
                aux_data={
                    "current_user_id": current_user.userId,
                    "chat_id": chat_id,
                    "context": query_data.context
                }
            )
            
            log_ai_response(ai_response, "/query")
            
        except Exception as e:
            log_error(f"AI processing error: {str(e)}")
            ai_response = "I apologize, but I'm experiencing technical difficulties. Please try again later."
        
        # Update query with response
        db_query.response = ai_response
        db.commit()
        
        # Update conversation history in short-term memory
        try:
            conversations = st_data.get("conversations", []) if isinstance(st_data, dict) else []
            
            # Simple conversation summary (without external handler)
            summary = f"Q: {query_data.query[:50]}... A: {ai_response[:50]}..."
            
            # Update long-term memory (simple approach)
            updated_lt = lt_data.copy() if isinstance(lt_data, dict) else {}
            
            # Add to conversation history
            conversations.append({
                "query": query_data.query,
                "response": ai_response,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "chat_id": chat_id
            })
            
            # Keep only recent conversations (last 50)
            if len(conversations) > 50:
                conversations = conversations[-50:]
            
            # Update short-term memory file
            if isinstance(st_data, dict):
                st_data["conversations"] = conversations
                shortterm_file.content = json.dumps(st_data, indent=2).encode('utf-8')
                shortterm_file.size = len(shortterm_file.content)
                shortterm_file.uploaded_at = datetime.now()
            
            # Update long-term memory file if changed
            if updated_lt and longterm_file:
                longterm_file.content = json.dumps(updated_lt, indent=2).encode('utf-8')
                longterm_file.size = len(longterm_file.content)
                longterm_file.uploaded_at = datetime.now()
            
            db.commit()
            
        except Exception as e:
            log_error(f"Memory update error: {str(e)}")
            # Continue even if memory update fails
        
        response_data = {
            "success": True,
            "query_id": db_query.queryId,
            "response": ai_response,
            "chat_id": chat_id,
            "timestamp": datetime.now().isoformat()
        }
        
        log_response(200, "Query processed successfully", "/query")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Query processing error: {str(e)}", e, endpoint="/query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )
