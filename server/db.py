"""Database Models Module for LMS Platform.

This module defines the database models and connection setup for the LMS platform.
It includes the User model and database connection configuration.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, LargeBinary, UniqueConstraint, SmallInteger, BigInteger, Boolean
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
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

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
    device_uuid = Column("device_uuid", String, unique=True, nullable=True)
    device_name = Column("device_name", String, nullable=False)
    device_type = Column("device_type", String, nullable=False)
    last_seen = Column("last_seen", DateTime, default=datetime.utcnow)
    online = Column("online", Boolean, default=False)

    # Relationship back to user
    user = relationship("User", back_populates="devices")
    """Relationship to the User who owns this device"""
    file_device_updates = relationship("FileDeviceUpdate", back_populates="device")
    """Relationship to FileDeviceUpdate objects for this device"""
    activities = relationship("DeviceActivity", back_populates="device")
    """Relationship to DeviceActivity objects for this device"""


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


class DeviceActivity(Base):
    """Model for tracking device activity and usage statistics."""
    __tablename__ = "device_activity"
    
    activity_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    device_id = Column(Integer, ForeignKey("device.device_id"), nullable=False)
    activity_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    activity_duration = Column(Integer, default=0)  # in seconds
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    device = relationship("Device", back_populates="activities")
    
    __table_args__ = (
        UniqueConstraint('device_id', 'activity_date', name='_device_date_uc'),
    )


# DO NOT run migrations or create tables at import time in serverless environments!
# Run this manually in a migration script or CLI, not here:
# Base.metadata.create_all(bind=engine)
