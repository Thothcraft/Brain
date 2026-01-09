"""Database Models Module for LMS Platform.

This module defines the database models and connection setup for the LMS platform.
It includes the User model and database connection configuration.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, LargeBinary, UniqueConstraint, SmallInteger, BigInteger, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://lms_user:lms_password@localhost:5432/thoth")

print(f"[DB] Using DATABASE_URL: {DATABASE_URL}")

# SQLAlchemy setup
Base = declarative_base()

# Configure engine with connection pooling optimized for Supabase/cloud PostgreSQL
# - pool_pre_ping: Test connections before use to detect stale connections
# - pool_recycle: Recycle connections frequently to prevent SSL timeouts
# - pool_size: Keep pool small to avoid connection limits
# - connect_args: Set connection timeout and keepalives
engine = create_engine(
    DATABASE_URL,
    pool_size=10,  # Increased base connections
    max_overflow=20,  # Increased overflow for parallel requests
    pool_timeout=60,  # Increased timeout
    pool_pre_ping=True,
    pool_recycle=600,  # Recycle connections every 10 minutes
    connect_args={
        "connect_timeout": 60,  # Increased for large file uploads
        "keepalives": 1,
        "keepalives_idle": 60,
        "keepalives_interval": 30,
        "keepalives_count": 10,
        "options": "-c statement_timeout=300000"  # 5 minute statement timeout
    }
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class User(Base):
    """User model representing registered users in the system.
    
    This model stores user credentials and settings. The password is stored
    as a hash, never in plain text.
    """
    __tablename__ = "user_account"
    userId = Column("user_id", Integer, primary_key=True, autoincrement=True, index=True)
    """Unique identifier for the user"""
    username = Column("username", String, unique=True, index=True, nullable=False)
    """Unique username for authentication"""
    hashed_password = Column("hashed_password", String)
    """Bcrypt hash of the user's password"""
    max_file_size = Column("max_file_size", Integer, default=524288000)  # 500MB default max file size
    """Maximum allowed file size in bytes (default: 500MB)"""
    role = Column("role", SmallInteger, default=0)  # Added user's role, int2 with default 0
    """User's role (int2 with default 0)"""
    phone_number = Column("phone_number", BigInteger, nullable=True, unique=True, index=True) # Added phone number
    """User's phone number"""
    
    # Relationships
    files = relationship("File", back_populates="user")
    """Relationship to File objects uploaded by this user"""
    queries = relationship("Query", back_populates="user")
    """Relationship to Query objects created by this user"""
    sessions = relationship("Session", back_populates="user")
    """Relationship to Session objects for this user"""
    devices = relationship("Device", back_populates="user")
    """Relationship to Device objects for this user"""


class File(Base):
    """File model representing files uploaded by users.
    
    Attributes:
        fileId: Unique identifier for the file
        filename: Name of the uploaded file
        userId: Foreign key to the user who uploaded the file
        path: Path where the file is stored on the server (nullable)
        size: Size of the file in bytes
        content: Binary content of the file
        content_type: MIME type of the file
        uploaded_at: Timestamp when the file was uploaded
        user: Relationship to the User who owns this file
    """
    __tablename__ = "file"
    fileId = Column("file_id", Integer, primary_key=True, autoincrement=True, index=True)
    """Unique identifier for the file"""
    filename = Column("file_name", String, nullable=False)
    """Name of the uploaded file"""
    userId = Column("user_id", Integer, ForeignKey("user_account.user_id"), nullable=False)
    """Foreign key to the user who uploaded the file"""
    path = Column("path", String, nullable=True)  # Now nullable since we store content in DB
    """Path where the file is stored on the server (nullable)"""
    size = Column("size", Integer, nullable=False)
    """Size of the file in bytes"""
    content = Column("content", LargeBinary, nullable=True)  # Binary content of the file
    """Binary content of the file"""
    content_type = Column("content_type", String(255), nullable=True)  # MIME type
    """MIME type of the file"""
    uploaded_at = Column("uploaded_at", DateTime, default=datetime.utcnow)
    """Timestamp when the file was uploaded"""
    file_hash = Column("file_hash", Text, nullable=True)
    """Hash of the file"""
    last_modified = Column("last_modified", DateTime, nullable=True)
    """Timestamp when the file was last modified"""
    
    # Relationships
    user = relationship("User", back_populates="files")
    """Relationship to the User who owns this file"""
    file_device_updates = relationship("FileDeviceUpdate", back_populates="file")
    """Relationship to FileDeviceUpdate objects for this file"""


class Query(Base):
    """Query model representing AI queries made by users.
    
    Attributes:
        queryId: Unique identifier for the query
        userId: Foreign key to the user who made the query
        chatId: Identifier for grouping related queries into conversations
        query_text: The text of the user's query
        response: The AI response to the query
        created_at: Timestamp when the query was made
        user: Relationship to the User who made this query
    """
    __tablename__ = "query"
    queryId = Column("query_id", Integer, primary_key=True, autoincrement=True, index=True)
    """Unique identifier for the query"""
    userId = Column("user_id", Integer, ForeignKey("user_account.user_id"), nullable=False)
    """Foreign key to the user who made the query"""
    chatId = Column("chat_id", String, nullable=True)
    """Identifier for grouping related queries into conversations"""
    query_text = Column("query_text", Text, nullable=False)
    """The text of the user's query"""
    response = Column("response", Text, nullable=True)
    """The AI response to the query"""
    created_at = Column("created_at", DateTime, default=datetime.utcnow)
    """Timestamp when the query was made"""
    
    # Relationships
    user = relationship("User", back_populates="queries")
    """Relationship to the User who made this query"""


class Session(Base):
    """Session model for tracking user login sessions.
    
    Attributes:
        sessionId: Unique identifier for the session
        userId: Foreign key to the user who owns this session
        token: Session token for authentication
        expires_at: Timestamp when the session expires
        user: Relationship to the User who owns this session
    """
    __tablename__ = "session"
    sessionId = Column("session_id", Integer, primary_key=True, autoincrement=True, index=True)
    """Unique identifier for the session"""
    userId = Column("user_id", Integer, ForeignKey("user_account.user_id"), nullable=False)
    """Foreign key to the user who owns this session"""
    token = Column("token", String, nullable=False)
    """Session token for authentication"""
    expires_at = Column("expires_at", DateTime, nullable=False)
    """Timestamp when the session expires"""
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    """Relationship to the User who owns this session"""



class Device(Base):
    """Device model representing user devices that interact with the platform."""
    __tablename__ = "device"

    deviceId = Column("device_id", Integer, primary_key=True, autoincrement=True, index=True)
    userId = Column("user_id", Integer, ForeignKey("user_account.user_id"), nullable=False)
    # Stable, client-provided identifier, e.g. IOPlatformUUID (mac) or extension ID (chrome)
    device_uuid = Column("device_uuid", String, unique=True, nullable=False, index=True)
    device_name = Column("device_name", String, nullable=False)
    device_type = Column("device_type", String, nullable=False, default="thoth")
    
    last_seen = Column("last_seen", DateTime, default=datetime.utcnow, index=True)
    online = Column("online", Boolean, default=False, index=True)
    
    ip_address = Column("ip_address", String, nullable=True)
    mac_address = Column("mac_address", String, nullable=True)
    battery_level = Column("battery_level", Integer, nullable=True)
    
    

    # Relationship back to user
    user = relationship("User", back_populates="devices")
    """Relationship to the User who owns this device"""
    file_device_updates = relationship("FileDeviceUpdate", back_populates="device")
    """Relationship to FileDeviceUpdate objects for this device"""
    
    def to_dict(self):
        return {
            "device_id": self.device_uuid,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "online": self.online,
            "battery_level": self.battery_level,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "device_uuid": self.device_uuid,
            "user_id": self.userId
        }



class FileDeviceUpdate(Base):
    """Model representing updates of files across devices."""
    __tablename__ = "file_device_update"

    updateId = Column("update_id", Integer, primary_key=True, autoincrement=True, index=True)
    fileId = Column("file_id", Integer, ForeignKey("file.file_id"), nullable=False)
    deviceId = Column("device_id", Integer, ForeignKey("device.device_id"), nullable=False)
    updated_at = Column("updated_at", DateTime, default=datetime.utcnow)
    file_hash = Column("file_hash", Text, nullable=True)

    # Relationships
    file = relationship("File", back_populates="file_device_updates")
    """Relationship to the File that this update is for"""
    device = relationship("Device", back_populates="file_device_updates")
    """Relationship to the Device that this update is for"""


class DeviceFile(Base):
    """Model representing files that exist on a device.
    
    Files start as on_device=True, on_cloud=False.
    When uploaded to cloud, on_cloud becomes True and cloud_file_id is set.
    """
    __tablename__ = "device_file"
    
    id = Column("id", Integer, primary_key=True, autoincrement=True, index=True)
    device_id = Column("device_id", Integer, ForeignKey("device.device_id"), nullable=False, index=True)
    user_id = Column("user_id", Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    filename = Column("filename", String, nullable=False)
    size = Column("size", BigInteger, nullable=True)
    file_type = Column("file_type", String, nullable=True)  # imu, csi, mfcw, img, vid, other
    created_at = Column("created_at", DateTime, nullable=True)
    modified_at = Column("modified_at", DateTime, nullable=True)
    on_device = Column("on_device", Boolean, default=True)
    on_cloud = Column("on_cloud", Boolean, default=False)
    cloud_file_id = Column("cloud_file_id", Integer, ForeignKey("file.file_id"), nullable=True)
    upload_requested = Column("upload_requested", Boolean, default=False)  # Set by Research Portal to request upload
    last_synced = Column("last_synced", DateTime, default=datetime.utcnow)
    
    # Unique constraint: one file per device
    __table_args__ = (
        UniqueConstraint('device_id', 'filename', name='uq_device_filename'),
    )
    
    # Relationships
    device = relationship("Device", backref="device_files")
    user = relationship("User")
    cloud_file = relationship("File")
    
    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "size": self.size,
            "file_type": self.file_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "on_device": self.on_device,
            "on_cloud": self.on_cloud,
            "cloud_file_id": self.cloud_file_id,
            "upload_requested": self.upload_requested,
            "last_synced": self.last_synced.isoformat() if self.last_synced else None,
        }


class TrainingDataset(Base):
    """Dataset for training - groups files with labels."""
    __tablename__ = "training_dataset"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    files = relationship("DatasetFile", back_populates="dataset", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "file_count": len(self.files) if self.files else 0,
            "labels": list(set(f.label for f in self.files)) if self.files else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DatasetFile(Base):
    """Links files to datasets with labels for training."""
    __tablename__ = "dataset_file"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    dataset_id = Column(Integer, ForeignKey("training_dataset.id", ondelete="CASCADE"), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey("file.file_id"), nullable=False, index=True)
    label = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # No unique constraint - allow same file with different labels for testing
    
    # Relationships
    dataset = relationship("TrainingDataset", back_populates="files")
    file = relationship("File")
    
    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "file_id": self.file_id,
            "filename": self.file.filename if self.file else None,
            "label": self.label,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrainingJob(Base):
    """Persists training jobs for cloud training."""
    __tablename__ = "training_job"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    job_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("training_dataset.id"), nullable=True)
    test_dataset_id = Column(Integer, ForeignKey("training_dataset.id"), nullable=True)
    model_type = Column(String(50), nullable=False)
    training_mode = Column(String(50), nullable=False)
    config = Column(Text, nullable=True)  # JSON string
    status = Column(String(50), default="pending")
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer)
    metrics = Column(Text, nullable=True)  # JSON string
    best_metrics = Column(Text, nullable=True)  # JSON string
    model_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User")
    dataset = relationship("TrainingDataset", foreign_keys=[dataset_id])
    test_dataset = relationship("TrainingDataset", foreign_keys=[test_dataset_id])
    
    def to_dict(self):
        import json
        return {
            "id": self.id,
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset.name if self.dataset else None,
            "test_dataset_id": self.test_dataset_id,
            "model_type": self.model_type,
            "training_mode": self.training_mode,
            "config": json.loads(self.config) if self.config else {},
            "status": self.status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "metrics": json.loads(self.metrics) if self.metrics else {},
            "best_metrics": json.loads(self.best_metrics) if self.best_metrics else {},
            "model_path": self.model_path,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TrainedModel(Base):
    """Stores completed trained models for deployment."""
    __tablename__ = "trained_model"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    job_id = Column(String(255), nullable=True)
    name = Column(String(255), nullable=False)
    architecture = Column(String(50), nullable=True)
    accuracy = Column(Float, nullable=True)  # Stored as percentage (0-100)
    size_bytes = Column(BigInteger, nullable=True)
    model_data = Column(LargeBinary, nullable=True)
    config = Column(Text, nullable=True)  # JSON string
    is_pinned = Column(Boolean, default=False)  # Pinned models won't be auto-deleted
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    
    def to_dict(self):
        import json
        return {
            "id": self.id,
            "job_id": self.job_id,
            "name": self.name,
            "architecture": self.architecture,
            "accuracy": round(self.accuracy, 2) if self.accuracy else None,
            "size_mb": self.size_bytes / (1024 * 1024) if self.size_bytes else None,
            "config": json.loads(self.config) if self.config else {},
            "is_pinned": self.is_pinned,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# DO NOT run migrations or create tables at import time in serverless environments!
# Run this manually in a migration script or CLI, not here:
# Base.metadata.create_all(bind=engine)
