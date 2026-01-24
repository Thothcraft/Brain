"""FL Participation Request System for Thoth Devices.

This module handles FL participation requests sent to Thoth devices,
allowing device owners to approve or reject participation in FL sessions.

Flow:
1. Brain server creates FL session with remote Thoth devices
2. Brain sends participation request to each selected device
3. Device shows notification to user asking for permission
4. User approves/rejects on the device
5. Device sends response back to Brain
6. If all required devices approve, FL session starts
7. Devices receive progress updates during training

Reference: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
"""

import asyncio
import logging
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

# Thread lock for concurrent access to participation manager
_participation_lock = threading.RLock()


class RequestStatus(str, Enum):
    """Status of an FL participation request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class FLParticipationRequest:
    """A request for a Thoth device to participate in FL."""
    request_id: str
    session_id: str
    session_name: str
    device_id: str
    
    # Session details for user to review
    algorithm: str
    dataset: str
    num_rounds: int
    estimated_duration_minutes: int
    data_samples_needed: int
    
    # Request status
    status: RequestStatus = RequestStatus.PENDING
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    
    # Response details
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "session_name": self.session_name,
            "device_id": self.device_id,
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "num_rounds": self.num_rounds,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "data_samples_needed": self.data_samples_needed,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
            "rejection_reason": self.rejection_reason,
        }
    
    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class FLProgressUpdate:
    """Progress update for an FL session sent to participating devices."""
    session_id: str
    device_id: str
    
    # Progress info
    current_round: int
    total_rounds: int
    status: str  # running, completed, failed, cancelled
    
    # Metrics
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    
    # Device-specific metrics
    device_accuracy: float = 0.0
    device_loss: float = 0.0
    device_contribution: float = 0.0
    
    # Timing
    round_start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Messages
    message: str = ""
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "session_id": self.session_id,
            "device_id": self.device_id,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "progress_percent": round((self.current_round / self.total_rounds) * 100, 1) if self.total_rounds > 0 else 0,
            "status": self.status,
            "global_accuracy": self.global_accuracy,
            "global_loss": self.global_loss,
            "device_accuracy": self.device_accuracy,
            "device_loss": self.device_loss,
            "device_contribution": self.device_contribution,
            "round_start_time": self.round_start_time.isoformat() if self.round_start_time else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class FLParticipationManager:
    """Manages FL participation requests and progress updates for Thoth devices.
    
    Thread-safe implementation for concurrent access from multiple API requests.
    """
    
    def __init__(self):
        # Requests indexed by request_id
        self.requests: Dict[str, FLParticipationRequest] = {}
        # Requests indexed by device_id for quick lookup
        self.device_requests: Dict[str, List[str]] = {}
        # Active sessions per device
        self.active_sessions: Dict[str, str] = {}  # device_id -> session_id
        # Progress updates per device
        self.progress_updates: Dict[str, FLProgressUpdate] = {}  # device_id -> latest update
        # Callbacks for when requests are responded to
        self._response_callbacks: Dict[str, Callable] = {}
        # Request expiry time in minutes
        self.request_expiry_minutes = 5
        # Statistics
        self._stats = {
            "total_requests_created": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "total_expired": 0,
        }
        logger.info("[FL Participation] Manager initialized")
    
    def create_request(
        self,
        session_id: str,
        session_name: str,
        device_id: str,
        algorithm: str,
        dataset: str,
        num_rounds: int,
        estimated_duration_minutes: int = 30,
        data_samples_needed: int = 0,
    ) -> FLParticipationRequest:
        """Create a new FL participation request for a device.
        
        Args:
            session_id: FL session ID
            session_name: Human-readable session name
            device_id: Target device ID
            algorithm: FL algorithm name
            dataset: Dataset name
            num_rounds: Number of FL rounds
            estimated_duration_minutes: Estimated training duration
            data_samples_needed: Minimum data samples needed
        
        Returns:
            Created FLParticipationRequest
        
        Raises:
            ValueError: If required parameters are invalid
        """
        # Validate inputs
        if not session_id or not device_id:
            raise ValueError("session_id and device_id are required")
        if num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        
        request_id = str(uuid.uuid4())
        
        request = FLParticipationRequest(
            request_id=request_id,
            session_id=session_id,
            session_name=session_name or f"Session-{session_id[:8]}",
            device_id=device_id,
            algorithm=algorithm or "fedavg",
            dataset=dataset or "unknown",
            num_rounds=num_rounds,
            estimated_duration_minutes=max(1, estimated_duration_minutes),
            data_samples_needed=max(0, data_samples_needed),
            expires_at=datetime.now() + timedelta(minutes=self.request_expiry_minutes),
        )
        
        with _participation_lock:
            self.requests[request_id] = request
            
            if device_id not in self.device_requests:
                self.device_requests[device_id] = []
            self.device_requests[device_id].append(request_id)
            
            self._stats["total_requests_created"] += 1
        
        logger.info(f"[FL Request] Created request {request_id[:8]} for device {device_id[:8]} to join session '{session_name}' ({algorithm}, {num_rounds} rounds)")
        
        return request
    
    def get_request(self, request_id: str) -> Optional[FLParticipationRequest]:
        """Get a request by ID."""
        return self.requests.get(request_id)
    
    def get_pending_requests(self, device_id: str) -> List[FLParticipationRequest]:
        """Get all pending requests for a device.
        
        Args:
            device_id: Device ID
        
        Returns:
            List of pending requests
        """
        request_ids = self.device_requests.get(device_id, [])
        pending = []
        
        for request_id in request_ids:
            request = self.requests.get(request_id)
            if request and request.status == RequestStatus.PENDING:
                # Check if expired
                if request.is_expired():
                    request.status = RequestStatus.EXPIRED
                else:
                    pending.append(request)
        
        return pending
    
    def approve_request(self, request_id: str) -> bool:
        """Approve an FL participation request.
        
        Args:
            request_id: Request ID
        
        Returns:
            True if approved, False if request not found or already responded
        """
        if not request_id:
            logger.warning("[FL Request] approve_request called with empty request_id")
            return False
        
        with _participation_lock:
            request = self.requests.get(request_id)
            if not request:
                logger.warning(f"[FL Request] Request {request_id[:8]} not found for approval")
                return False
            
            if request.status != RequestStatus.PENDING:
                logger.warning(f"[FL Request] Request {request_id[:8]} already has status: {request.status.value}")
                return False
            
            if request.is_expired():
                request.status = RequestStatus.EXPIRED
                self._stats["total_expired"] += 1
                logger.warning(f"[FL Request] Request {request_id[:8]} has expired")
                return False
            
            request.status = RequestStatus.APPROVED
            request.responded_at = datetime.now()
            
            # Mark device as active in this session
            self.active_sessions[request.device_id] = request.session_id
            self._stats["total_approved"] += 1
        
        logger.info(f"[FL Request] Request {request_id[:8]} APPROVED by device {request.device_id[:8]} for session {request.session_id[:8]}")
        
        # Trigger callback if registered (outside lock to avoid deadlocks)
        if request_id in self._response_callbacks:
            try:
                self._response_callbacks[request_id](request)
            except Exception as e:
                logger.error(f"[FL Request] Error in approval callback for {request_id[:8]}: {e}", exc_info=True)
        
        return True
    
    def reject_request(self, request_id: str, reason: Optional[str] = None) -> bool:
        """Reject an FL participation request.
        
        Args:
            request_id: Request ID
            reason: Optional rejection reason
        
        Returns:
            True if rejected, False if request not found or already responded
        """
        if not request_id:
            logger.warning("[FL Request] reject_request called with empty request_id")
            return False
        
        with _participation_lock:
            request = self.requests.get(request_id)
            if not request:
                logger.warning(f"[FL Request] Request {request_id[:8]} not found for rejection")
                return False
            
            if request.status != RequestStatus.PENDING:
                logger.warning(f"[FL Request] Request {request_id[:8]} already has status: {request.status.value}")
                return False
            
            request.status = RequestStatus.REJECTED
            request.responded_at = datetime.now()
            request.rejection_reason = reason
            self._stats["total_rejected"] += 1
        
        logger.info(f"[FL Request] Request {request_id[:8]} REJECTED by device {request.device_id[:8]}: {reason or 'No reason provided'}")
        
        # Trigger callback if registered (outside lock)
        if request_id in self._response_callbacks:
            try:
                self._response_callbacks[request_id](request)
            except Exception as e:
                logger.error(f"[FL Request] Error in rejection callback for {request_id[:8]}: {e}", exc_info=True)
        
        return True
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an FL participation request (from server side).
        
        Args:
            request_id: Request ID
        
        Returns:
            True if cancelled, False if request not found
        """
        request = self.requests.get(request_id)
        if not request:
            return False
        
        if request.status == RequestStatus.PENDING:
            request.status = RequestStatus.CANCELLED
            logger.info(f"[FL Request] Request {request_id[:8]} cancelled")
        
        return True
    
    def register_response_callback(self, request_id: str, callback: Callable) -> None:
        """Register a callback to be called when a request is responded to.
        
        Args:
            request_id: Request ID
            callback: Callback function that takes the request as argument
        """
        self._response_callbacks[request_id] = callback
    
    def send_progress_update(
        self,
        session_id: str,
        device_id: str,
        current_round: int,
        total_rounds: int,
        status: str,
        global_accuracy: float = 0.0,
        global_loss: float = 0.0,
        device_accuracy: float = 0.0,
        device_loss: float = 0.0,
        device_contribution: float = 0.0,
        message: str = "",
        estimated_completion: Optional[datetime] = None,
    ) -> FLProgressUpdate:
        """Send a progress update to a device.
        
        Args:
            session_id: FL session ID
            device_id: Target device ID
            current_round: Current round number
            total_rounds: Total number of rounds
            status: Session status
            global_accuracy: Global model accuracy
            global_loss: Global model loss
            device_accuracy: Device's contribution accuracy
            device_loss: Device's contribution loss
            device_contribution: Device's contribution score
            message: Optional message
            estimated_completion: Estimated completion time
        
        Returns:
            Created FLProgressUpdate
        """
        update = FLProgressUpdate(
            session_id=session_id,
            device_id=device_id,
            current_round=current_round,
            total_rounds=total_rounds,
            status=status,
            global_accuracy=global_accuracy,
            global_loss=global_loss,
            device_accuracy=device_accuracy,
            device_loss=device_loss,
            device_contribution=device_contribution,
            message=message,
            estimated_completion=estimated_completion,
            round_start_time=datetime.now(),
        )
        
        self.progress_updates[device_id] = update
        
        return update
    
    def get_progress_update(self, device_id: str) -> Optional[FLProgressUpdate]:
        """Get the latest progress update for a device.
        
        Args:
            device_id: Device ID
        
        Returns:
            Latest FLProgressUpdate or None
        """
        return self.progress_updates.get(device_id)
    
    def get_active_session(self, device_id: str) -> Optional[str]:
        """Get the active FL session for a device.
        
        Args:
            device_id: Device ID
        
        Returns:
            Session ID or None
        """
        return self.active_sessions.get(device_id)
    
    def end_device_session(self, device_id: str) -> None:
        """End the active FL session for a device.
        
        Args:
            device_id: Device ID
        """
        if device_id in self.active_sessions:
            del self.active_sessions[device_id]
        if device_id in self.progress_updates:
            del self.progress_updates[device_id]
    
    def cleanup_expired_requests(self) -> int:
        """Clean up expired requests.
        
        Returns:
            Number of requests cleaned up
        """
        cleaned = 0
        for request_id, request in list(self.requests.items()):
            if request.status == RequestStatus.PENDING and request.is_expired():
                request.status = RequestStatus.EXPIRED
                cleaned += 1
        
        return cleaned


# Global instance
fl_participation_manager = FLParticipationManager()
