"""
Logging Utilities Module

This module provides specialized logging functions to standardize
and enhance the logging across the entire server application.
"""

import logging
import os
import sys
import platform
import psutil
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Union
from flask import request

# Set up logger (guarded to avoid duplicate handlers on reload/worker spawn)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent double logging via root/uvicorn handlers

if not logger.handlers:
    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

def get_logger(name: str = None) -> logging.Logger:
    """Return a configured logger with duplicate-handler protection.

    Args:
        name: Optional child logger name.
    """
    if not name:
        return logger
    child = logging.getLogger(name)
    child.propagate = False
    if not child.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        child.addHandler(ch)
    return child

def log_server_lifecycle(event: str, details: Optional[Dict] = None) -> None:
    """Log server lifecycle events with detailed information.
    
    Args:
        event (str): The lifecycle event (startup, shutdown, etc.)
        details (dict, optional): Additional details about the event
    """
    logger.info(f"[SERVER] Lifecycle event: {event}")
    logger.info(f"[SERVER] Timestamp: {datetime.now().isoformat()}")
    if details:
        logger.info(f"[SERVER] Event details: {details}")

def log_server_health() -> None:
    """Log server health metrics including CPU, memory, and threads."""
    process = psutil.Process()
    logger.info(f"[SERVER] Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    logger.info(f"[SERVER] Thread count: {process.num_threads()}")
    logger.info(f"[SERVER] Process start time: {datetime.fromtimestamp(process.create_time()).isoformat()}")
    logger.info(f"[SERVER] System uptime: {datetime.fromtimestamp(psutil.boot_time()).isoformat()}")

def log_request_start(endpoint: str, method: str = 'UNKNOWN', 
                    request=None, remote_addr: Optional[str] = None, 
                    user_id: Optional[int] = None):
    """Log the start of a request with detailed information.
    
    Args:
        endpoint (str): The API endpoint being accessed
        method (str): The HTTP method used
        request: The FastAPI Request object or None
        remote_addr (str, optional): The client's IP address
        user_id (int, optional): The ID of the authenticated user
    """
    try:
        if request and hasattr(request, 'get'):
            # Handle ASGI scope
            headers = dict(request.get('headers', []))
            method = request.get('method', method)
            remote_addr = remote_addr or request.get('client', ['unknown'])[0]
        elif request and hasattr(request, 'headers'):
            # Handle FastAPI Request object
            headers = dict(request.headers)
            method = request.method
            remote_addr = remote_addr or request.client.host if request.client else 'unknown'
        else:
            headers = {}
            
        logger.info(
            "[REQUEST] %s %s from %s (User ID: %s)\nHeaders: %s",
            method,
            endpoint,
            remote_addr,
            user_id or 'unauthenticated',
            {k: v for k, v in headers.items() if k.lower() not in ['authorization', 'cookie']}
        )
    except Exception as e:
        logger.error("Error in log_request_start: %s", str(e))

def log_request_payload(payload: Any, endpoint: str) -> None:
    """Log details about the request payload.
    
    Args:
        endpoint: The API endpoint being accessed
    """
    if payload:
        try:
            # Safely convert payload to string, handling non-serializable objects
            if hasattr(payload, 'dict'):
                payload_str = str(payload.dict())
            elif isinstance(payload, (str, bytes)):
                payload_str = str(payload)[:500]  # Truncate long strings
            else:
                payload_str = str(payload)[:500]
                
            logger.debug("[PAYLOAD] %s: %s", endpoint, payload_str)
        except Exception as e:
            logger.warning("[PAYLOAD] Failed to log payload for %s: %s", endpoint, str(e))

def log_validation(field: str, value: Any, valid: bool, endpoint: str) -> None:
    """Log field validation results.
    
    Args:
        field (str): The field being validated
        value (any): The value being validated (will be truncated if string)
        valid (bool): Whether validation passed
        endpoint (str): The API endpoint being accessed
    """
    value_str = str(value)
    if len(value_str) > 100:  # Truncate long values
        value_str = value_str[:100] + "..."
    
    if valid:
        logger.debug("[VALID] %s: %s is valid", field, value_str)
    else:
        logger.warning("[VALID] %s: %s is invalid", field, value_str)

def log_error(msg: str, exc: Optional[Exception] = None, context: Optional[Dict] = None, 
              endpoint: Optional[str] = None) -> None:
    """Log an error with optional exception and context.
    
    Args:
        msg: Error message
        exc: Optional exception that caused the error
        context: Additional context about the error
        endpoint: Optional API endpoint where the error occurred
    """
    log_msg = f"Error in {endpoint}: {msg}" if endpoint else f"Error: {msg}"
    if context:
        log_msg += f"\nContext: {context}"
    
    if exc:
        logger.exception(log_msg, exc_info=exc)
    else:
        logger.error(log_msg)


def log_response(status_code: int, response: Any, endpoint: str) -> None:
    """Log the response being sent back to the client.
    
    Args:
        status_code: HTTP status code
        response: The response data
        endpoint: The API endpoint that was called
    """
    log_level = logging.INFO if status_code < 400 else logging.ERROR
    logger.log(
        log_level,
        "Response from %s - Status: %s, Response: %s",
        endpoint, status_code, response
    )


def log_ai_call(query: str, model: str, endpoint: str) -> None:
    """Log when an AI query is made.
    
    Args:
        query: The user's query
        model: The AI model being used
        endpoint: The API endpoint handling the query
    """
    logger.info(
        "AI Query - Endpoint: %s, Model: %s, Query: %s",
        endpoint, model, query
    )


def log_ai_response(response: str, endpoint: str) -> None:
    """Log the AI's response.
    
    Args:
        response: The AI's response
        endpoint: The API endpoint that handled the query
    """
    logger.debug("AI Response from %s: %s", endpoint, response)


def log_something(something: Any, endpoint: str) -> None:
    """Generic logging function for miscellaneous information.
    
    Args:
        something: The thing to log
        endpoint: The API endpoint where this is being logged from
    """
    logger.info("[MISC] %s - %s: %s", endpoint, type(something).__name__, str(something)[:200])

def log_file_operation(operation: str, file_path: str, success: bool, 
                     details: Optional[Dict[str, Any]] = None, 
                     endpoint: Optional[str] = None) -> None:
    """Log file operations with details.
    
    Args:
        operation: Type of file operation (e.g., 'read', 'write', 'delete')
        file_path: Path to the file being operated on
        success: Whether the operation was successful
        details: Additional details about the operation
        endpoint: Optional API endpoint where the operation was initiated
    """
    status = "succeeded" if success else "failed"
    log_msg = f"[FILE] Operation '{operation}' {status} for file: {file_path}"
    
    if endpoint:
        log_msg = f"[{endpoint}] {log_msg}"
    
    if details:
        log_msg = f"{log_msg} - Details: {details}"
    
    if success:
        logger.info(log_msg)
    else:
        logger.error(log_msg)
