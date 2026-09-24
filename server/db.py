"""Database Models Module for LMS Platform.

This module defines the database models and connection setup for the LMS platform.
It includes the User model and database connection configuration.
"""

import os
import json
import time
import logging
from datetime import datetime
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, LargeBinary, UniqueConstraint, SmallInteger, BigInteger, Boolean, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError

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

# Import Pro Plan configuration
try:
    from server.pro_config import pro_config
    DATABASE_URL = pro_config.get_database_url()
    print(f"[DB] Using Supabase Pro Plan configuration")
except (ImportError, ValueError) as e:
    print(f"[DB] Pro Plan config not available, using standard configuration: {e}")

print("[DB] Database URL configured")

# SQLAlchemy setup
Base = declarative_base()

# Configure engine with connection pooling optimized for Supabase Pro Plan
try:
    from server.pro_config import pro_config
    engine_kwargs = pro_config.get_engine_kwargs()
    print(f"[DB] Using Pro Plan engine settings: pool_size={engine_kwargs['pool_size']}, max_overflow={engine_kwargs['max_overflow']}")
except ImportError:
    # Fallback to manual configuration
    engine_kwargs = {
        "pool_size": 20,  # Reduced from 50 to prevent connection exhaustion
        "max_overflow": 30,  # Reduced from 100 for better stability
        "pool_timeout": 30,
        "pool_pre_ping": True,
        "pool_recycle": 600,  # Reduced from 1800 to 10 minutes - more frequent recycling prevents SSL timeouts
        "pool_use_lifo": True,  # Use LIFO to recycle newer connections first
        "connect_args": {
            "connect_timeout": 10,  # Reduced from 30 for faster failure detection
            "application_name": "thoth_pro",
            "sslmode": "require",
            "keepalives": 1,
            "keepalives_idle": 15,  # Reduced from 30 to send keepalives more frequently
            "keepalives_interval": 5,  # Reduced from 10 for more frequent checks
            "keepalives_count": 3,  # Reduced from 5 to fail faster
            "options": "-c statement_timeout=30000 -c idle_in_transaction_session_timeout=60000 -c tcp_keepalives_idle=15 -c tcp_keepalives_interval=5 -c tcp_keepalives_count=3"
        }
    }
    print(f"[DB] Using fallback engine settings")

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# Configure logging for database issues
logging.basicConfig(level=logging.INFO)
db_logger = logging.getLogger('database')

def get_db():
    """Dependency to get database session with retry logic."""
    max_retries = 3
    retry_delay = 1.0  # Increased from 0.5 for better recovery
    
    for attempt in range(max_retries):
        db = SessionLocal()
        try:
            # Test the connection with a simple query
            db.execute(text("SELECT 1"))
            yield db
            return
        except (SQLAlchemyError, DisconnectionError, OperationalError) as e:
            db_logger.warning(f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}")
            try:
                db.close()
            except:
                pass
            
            # On SSL errors, dispose the engine to force new connections
            if "SSL" in str(e) or "closed unexpectedly" in str(e) or "server closed the connection unexpectedly" in str(e):
                db_logger.warning("SSL/connection error detected, disposing engine to force new connections")
                try:
                    engine.dispose()
                except:
                    pass
            
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                import random
                jitter = random.uniform(0.8, 1.2)
                time.sleep(retry_delay * (attempt + 1) * jitter)
            else:
                db_logger.error(f"All {max_retries} database connection attempts failed")
                raise
        finally:
            try:
                db.close()
            except:
                pass

@contextmanager
def get_db_session():
    """Context manager for database sessions with automatic cleanup."""
    max_retries = 3
    retry_delay = 1.0  # Increased from 0.5
    
    for attempt in range(max_retries):
        db = SessionLocal()
        try:
            # Test the connection
            db.execute(text("SELECT 1"))
            yield db
            db.commit()
            return
        except (SQLAlchemyError, DisconnectionError, OperationalError) as e:
            db_logger.warning(f"Database session attempt {attempt + 1}/{max_retries} failed: {e}")
            try:
                db.rollback()
            except:
                pass
            try:
                db.close()
            except:
                pass
            
            # On SSL errors, dispose the engine to force new connections
            if "SSL" in str(e) or "closed unexpectedly" in str(e) or "server closed the connection unexpectedly" in str(e):
                db_logger.warning("SSL/connection error detected, disposing engine")
                try:
                    engine.dispose()
                except:
                    pass
            
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                import random
                jitter = random.uniform(0.8, 1.2)
                time.sleep(retry_delay * (attempt + 1) * jitter)
            else:
                db_logger.error(f"All {max_retries} database session attempts failed")
                raise
        except Exception as e:
            try:
                db.rollback()
            except:
                pass
            try:
                db.close()
            except:
                pass
            raise
        finally:
            try:
                db.close()
            except:
                pass

def test_database_connection():
    """Test database connectivity and return status."""
    try:
        # Create a direct connection for testing (avoiding pool issues)
        direct_engine = create_engine(
            DATABASE_URL,
            pool_size=1,
            max_overflow=0,
            pool_timeout=5,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 5,
                "sslmode": "require",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
                "options": "-c statement_timeout=5000"
            }
        )
        
        with direct_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            
        return {
            "status": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "pool_size": engine.pool.size(),
            "checked_in": engine.pool.checkedin(),
            "checked_out": engine.pool.checkedout()
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

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
    role = Column("role", SmallInteger, default=0)  # 0=user, 1=admin, 2=organization
    """User's role: 0=regular user, 1=admin, 2=organization"""
    phone_number = Column("phone_number", BigInteger, nullable=True, unique=True, index=True)
    """User's phone number"""
    plan = Column("plan", String(50), default="free")  # free, researcher, organization
    """User's subscription plan"""
    stripe_customer_id = Column("stripe_customer_id", String(255), nullable=True)
    """Stripe customer ID"""
    stripe_subscription_id = Column("stripe_subscription_id", String(255), nullable=True)
    """Stripe subscription ID"""
    plan_expires_at = Column("plan_expires_at", DateTime, nullable=True)
    """When the current plan expires"""
    org_name = Column("org_name", String(255), nullable=True)
    """Organization display name (for role=2 accounts)"""
    email = Column("email", String(320), nullable=True, unique=True, index=True)
    """Verified contact email used for Supabase Auth and Stripe."""
    email_verified = Column("email_verified", Boolean, nullable=False, default=False)
    supabase_auth_user_id = Column("supabase_auth_user_id", String(36), nullable=True, unique=True, index=True)
    
    # Relationships
    files = relationship("File", back_populates="user")
    """Relationship to File objects uploaded by this user"""
    queries = relationship("Query", back_populates="user")
    """Relationship to Query objects created by this user"""
    sessions = relationship("Session", back_populates="user")
    """Relationship to Session objects for this user"""
    devices = relationship("Device", back_populates="user")
    """Relationship to Device objects for this user"""
    org_memberships_as_member = relationship("OrgMembership", foreign_keys="OrgMembership.member_id", back_populates="member")
    """Org memberships where this user is a member"""
    org_memberships_as_org = relationship("OrgMembership", foreign_keys="OrgMembership.org_id", back_populates="org")
    """Org memberships this org owns"""


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
    userId = Column("user_id", Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    """Foreign key to the user who uploaded the file"""
    path = Column("path", String, nullable=True)  # Now nullable since we store content in DB
    """Path where the file is stored on the server (nullable)"""
    size = Column("size", Integer, nullable=False)
    """Size of the file in bytes"""
    content = Column("content", LargeBinary, nullable=True)  # Binary content of the file
    """Binary content of the file"""
    content_type = Column("content_type", String(255), nullable=True)  # MIME type
    """MIME type of the file"""
    uploaded_at = Column("uploaded_at", DateTime, default=datetime.utcnow, index=True)
    """Timestamp when the file was uploaded"""
    file_hash = Column("file_hash", Text, nullable=True)
    """Hash of the file"""
    last_modified = Column("last_modified", DateTime, nullable=True)
    """Timestamp when the file was last modified"""
    storage_path = Column("storage_path", String(500), nullable=True)
    """Path in Supabase Storage (e.g., 'files/user_123/file_456/name.csv')"""
    sample_content = Column("sample_content", Text, nullable=True)
    """First few lines of the file for quick preview (max ~10KB)"""
    data_type = Column("data_type", String(50), nullable=True)
    """Detected data type: 'csi', 'imu', 'sensor', etc."""
    folder_id = Column("folder_id", Integer, ForeignKey("folder.folder_id"), nullable=True, index=True)
    """Foreign key to the folder containing this file (nullable for root files)"""
    labels = Column("labels", Text, nullable=True)
    """JSON array of labels assigned to this file"""
    
    # Relationships
    user = relationship("User", back_populates="files")
    """Relationship to the User who owns this file"""
    file_device_updates = relationship("FileDeviceUpdate", back_populates="file")
    """Relationship to FileDeviceUpdate objects for this file"""
    folder = relationship("Folder", back_populates="files")
    """Relationship to the Folder containing this file"""


class Folder(Base):
    """Folder model for organizing files.
    
    Attributes:
        folderId: Unique identifier for the folder
        name: Name of the folder
        userId: Foreign key to the user who owns the folder
        parent_id: Foreign key to parent folder (nullable for root folders)
        created_at: Timestamp when the folder was created
        updated_at: Timestamp when the folder was last updated
    """
    __tablename__ = "folder"
    folderId = Column("folder_id", Integer, primary_key=True, autoincrement=True, index=True)
    """Unique identifier for the folder"""
    name = Column("name", String(255), nullable=False)
    """Name of the folder"""
    userId = Column("user_id", Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    """Foreign key to the user who owns the folder"""
    parent_id = Column("parent_id", Integer, ForeignKey("folder.folder_id"), nullable=True, index=True)
    """Foreign key to parent folder (nullable for root folders)"""
    created_at = Column("created_at", DateTime, default=datetime.utcnow)
    """Timestamp when the folder was created"""
    updated_at = Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    """Timestamp when the folder was last updated"""
    description = Column("description", Text, nullable=True)
    """Optional description of the folder"""
    
    # Relationships
    user = relationship("User", backref="folders")
    """Relationship to the User who owns this folder"""
    files = relationship("File", back_populates="folder")
    """Relationship to files in this folder"""
    children = relationship("Folder", backref="parent", remote_side=[folderId])
    """Relationship to child folders"""
    
    def to_dict(self):
        return {
            "id": self.folderId,
            "name": self.name,
            "parent_id": self.parent_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "description": self.description,
        }


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
    userId = Column("user_id", Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    # Stable, client-provided identifier, e.g. IOPlatformUUID (mac) or extension ID (chrome)
    device_uuid = Column("device_uuid", String, unique=True, nullable=False, index=True)
    device_name = Column("device_name", String, nullable=False)
    device_type = Column("device_type", String, nullable=False, default="thoth")
    
    last_seen = Column("last_seen", DateTime, default=datetime.utcnow, index=True)
    online = Column("online", Boolean, default=False, index=True)
    approved = Column("approved", Boolean, default=False, index=True)
    
    ip_address = Column("ip_address", String, nullable=True)
    mac_address = Column("mac_address", String, nullable=True)
    battery_level = Column("battery_level", Integer, nullable=True)
    hardware_info = Column("hardware_info", Text, nullable=True)  # JSON string with device type, sensors, etc.
    

    # Relationship back to user
    user = relationship("User", back_populates="devices")
    """Relationship to the User who owns this device"""
    file_device_updates = relationship("FileDeviceUpdate", back_populates="device")
    """Relationship to FileDeviceUpdate objects for this device"""
    
    def to_dict(self):
        import json
        hw_info = None
        if self.hardware_info:
            try:
                hw_info = json.loads(self.hardware_info)
            except (json.JSONDecodeError, TypeError):
                hw_info = None
        
        # A device is online if it sent a heartbeat/register recently. The
        # dashboard polls through multiple services, so short windows make
        # active Thoth devices appear offline between sync cycles.
        if self.last_seen:
            age = (datetime.utcnow() - self.last_seen).total_seconds()
            is_online = age <= 900
            last_seen = self.last_seen.isoformat() + "Z"
        else:
            is_online = False
            last_seen = None

        return {
            "device_id": self.device_uuid,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "online": is_online,
            "battery_level": self.battery_level,
            "last_seen": last_seen,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "device_uuid": self.device_uuid,
            "user_id": self.userId,
            "hardware_info": hw_info,
            "portal_upload_allowed": bool(hw_info.get("portal_upload_allowed", True)) if isinstance(hw_info, dict) else True,
            "deployment_requests_allowed": bool(hw_info.get("deployment_requests_allowed", True)) if isinstance(hw_info, dict) else True,
            "cloud_sync_allowed": bool(hw_info.get("cloud_sync_allowed", True)) if isinstance(hw_info, dict) else True,
            "collection_active": bool(hw_info.get("collection_active", False)) if isinstance(hw_info, dict) else False,
            "approved": self.approved if self.approved is not None else False
        }


class DevicePairing(Base):
    """Short-lived handshake used to bind a physical Thoth to thothHUB."""
    __tablename__ = "device_pairing"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_uuid = Column(String(255), nullable=False, index=True)
    device_name = Column(String(255), nullable=False)
    device_type = Column(String(50), nullable=False, default="thoth")
    hardware_info = Column(Text, nullable=True)
    code_hash = Column(String(64), nullable=False, unique=True, index=True)
    secret_hash = Column(String(64), nullable=False, unique=True, index=True)
    status = Column(String(20), nullable=False, default="pending", index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id", ondelete="CASCADE"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False, index=True)
    claimed_at = Column(DateTime, nullable=True)



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
    upload_requested = Column("upload_requested", Boolean, default=False)  # Set by thothHUB to request upload
    last_synced = Column("last_synced", DateTime, default=datetime.utcnow)
    metadata_json = Column("metadata_json", Text, nullable=True)
    
    # Unique constraint: one file per device
    __table_args__ = (
        UniqueConstraint('device_id', 'filename', name='uq_device_filename'),
    )
    
    # Relationships
    device = relationship("Device", backref="device_files")
    user = relationship("User")
    cloud_file = relationship("File")
    
    def to_dict(self):
        def _utc(value):
            return value.isoformat() + "Z" if value else None

        try:
            metadata = json.loads(self.metadata_json or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        return {
            "id": self.id,
            "filename": self.filename,
            "size": self.size,
            "file_type": self.file_type,
            "created_at": _utc(self.created_at),
            "modified_at": _utc(self.modified_at),
            "on_device": self.on_device,
            "on_cloud": self.on_cloud,
            "cloud_file_id": self.cloud_file_id,
            "upload_requested": self.upload_requested,
            "last_synced": _utc(self.last_synced),
            "metadata": metadata,
        }


class DeviceCaptureChunk(Base):
    """Current-minute live analysis, optimized for frequent idempotent upserts."""
    __tablename__ = "device_capture_chunk"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("device.device_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id", ondelete="CASCADE"), nullable=False, index=True)
    minute = Column(String(13), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False, default="loading")
    occupied = Column(Boolean, nullable=True)
    frame_count = Column(Integer, nullable=False, default=10)
    payload = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)

    __table_args__ = (
        UniqueConstraint("device_id", "minute", "chunk_index", name="uq_device_capture_chunk"),
    )

    def to_dict(self):
        try:
            payload = json.loads(self.payload or "{}")
        except (TypeError, json.JSONDecodeError):
            payload = {}
        return {
            **payload,
            "minute": self.minute,
            "second_index": self.chunk_index,
            "chunk_index": self.chunk_index,
            "status": self.status,
            "occupied": self.occupied,
            "frame_count": self.frame_count,
            "updated_at": self.updated_at.isoformat() + "Z" if self.updated_at else None,
        }


class DeviceCommand(Base):
    """Durable portal/assistant command claimed by a device heartbeat."""
    __tablename__ = "device_command"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("device.device_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id", ondelete="CASCADE"), nullable=False, index=True)
    command = Column(String(40), nullable=False)
    payload = Column(Text, nullable=False, default="{}")
    status = Column(String(20), nullable=False, default="pending", index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    result = Column(Text, nullable=True)

    def to_dict(self):
        try:
            payload = json.loads(self.payload or "{}")
        except (TypeError, json.JSONDecodeError):
            payload = {}
        return {
            "id": self.id,
            "command": self.command,
            "payload": payload,
            "status": self.status,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }


class DeviceCapture(Base):
    """A durable logical capture session owned by exactly one device.

    Unlike ``DeviceCaptureChunk`` (per-minute live analysis), this row is the
    authoritative record of a capture request: it pins the capture to the
    device that must run it and tracks a real lifecycle state
    (requested → running → stopping → stopped/failed). This lets stop target
    the correct device and prevents reporting unconfirmed work as done.
    """
    __tablename__ = "device_capture"

    id = Column(Integer, primary_key=True, autoincrement=True)
    capture_id = Column(String(64), unique=True, nullable=False, index=True)
    device_id = Column(Integer, ForeignKey("device.device_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id", ondelete="CASCADE"), nullable=False, index=True)
    state = Column(String(20), nullable=False, default="requested", index=True)
    sensors = Column(Text, nullable=False, default="[]")
    sample_counts = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        try:
            sensors = json.loads(self.sensors or "[]")
        except (TypeError, json.JSONDecodeError):
            sensors = []
        try:
            counts = json.loads(self.sample_counts or "{}")
        except (TypeError, json.JSONDecodeError):
            counts = {}
        return {
            "id": self.capture_id,
            "device_id": self.device_id,
            "state": self.state,
            "sensors": sensors,
            "sample_counts": counts,
            "started_at": self.started_at.isoformat() + "Z" if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() + "Z" if self.stopped_at else None,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }


class AutomationKey(Base):
    """A scoped automation credential for programmatic v1 access.

    The raw key is shown once at creation; only its SHA-256 hash is stored.
    ``scopes`` is a JSON list such as ``["sensor:stream", "model:deploy"]``
    that gates which operations the key may perform (§9.4 / §17).
    """
    __tablename__ = "automation_key"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(120), nullable=False, default="")
    scopes = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    revoked = Column(Boolean, nullable=False, default=False, index=True)

    def scope_list(self):
        try:
            return list(json.loads(self.scopes or "[]"))
        except (TypeError, json.JSONDecodeError):
            return []

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "scopes": self.scope_list(),
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() + "Z" if self.last_used_at else None,
            "revoked": bool(self.revoked),
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
    storage_path = Column("storage_path", String(500), nullable=True)
    """Path in Supabase Storage (e.g., 'models/user_123/model_456/name.pt')"""
    config = Column(Text, nullable=True)  # JSON string
    is_pinned = Column(Boolean, default=False)  # Pinned models won't be auto-deleted
    # Processor-ecosystem metadata (rule | classical | torchscript | fusion)
    processor_type = Column(String(20), nullable=False, default="torchscript")
    sensor = Column(String(50), nullable=True)        # radar | csi | camera | fusion | any
    task = Column(String(50), nullable=True)          # occupancy | har | localization | environmental
    visibility = Column(String(20), nullable=False, default="private")  # private | community | official
    registry_name = Column(String(255), nullable=True, index=True)  # e.g. "thothcraft/radar-occupancy-v2"
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
            "processor_type": self.processor_type,
            "sensor": self.sensor,
            "task": self.task,
            "visibility": self.visibility,
            "registry_name": self.registry_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DeviceDeployment(Base):
    """Pending model deployments for pull-based delivery to devices."""
    __tablename__ = "device_deployment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    deployment_id = Column(String(255), unique=True, nullable=False, index=True)
    device_uuid = Column(String(255), nullable=False, index=True)
    model_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    payload = Column(Text, nullable=False)  # JSON: full deploy payload (model_data included)
    status = Column(String(50), default="pending", index=True)  # pending | delivered | failed
    created_at = Column(DateTime, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)


class Space(Base):
    """A named physical area (room, floor, building) with an optional
    floor plan. Devices are placed inside spaces; zones subdivide them."""
    __tablename__ = "space"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    parent_id = Column(Integer, ForeignKey("space.id"), nullable=True, index=True)  # building→floor→room
    name = Column(String(255), nullable=False)
    floor_plan_file_id = Column(Integer, ForeignKey("file.file_id"), nullable=True)
    width_m = Column(Float, nullable=True)   # plan extents in meters
    height_m = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    zones = relationship("Zone", back_populates="space", cascade="all, delete-orphan")
    placements = relationship("DevicePlacement", back_populates="space", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "floor_plan_file_id": self.floor_plan_file_id,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "zones": [z.to_dict() for z in self.zones],
            "devices": [p.to_dict() for p in self.placements],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Zone(Base):
    """A polygonal region inside a Space ('desk', 'bed')."""
    __tablename__ = "zone"

    id = Column(Integer, primary_key=True, autoincrement=True)
    space_id = Column(Integer, ForeignKey("space.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    polygon_json = Column(Text, nullable=False)  # JSON [[x,y],...] in meters
    created_at = Column(DateTime, default=datetime.utcnow)

    space = relationship("Space", back_populates="zones")

    def to_dict(self):
        import json as _json
        try:
            polygon = _json.loads(self.polygon_json)
        except (TypeError, ValueError):
            polygon = []
        return {"id": self.id, "space_id": self.space_id, "name": self.name,
                "polygon": polygon}


class DevicePlacement(Base):
    """Where a device sits in a Space and what its sensors cover."""
    __tablename__ = "device_placement"

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(Integer, ForeignKey("device.device_id", ondelete="CASCADE"),
                       unique=True, nullable=False, index=True)
    space_id = Column(Integer, ForeignKey("space.id", ondelete="CASCADE"), nullable=False, index=True)
    x = Column(Float, nullable=False, default=0.0)        # meters, plan coords
    y = Column(Float, nullable=False, default=0.0)
    rotation_deg = Column(Float, nullable=False, default=0.0)  # facing direction
    fov_deg = Column(Float, nullable=False, default=90.0)      # sensor cone
    range_m = Column(Float, nullable=False, default=8.0)       # sensor reach
    created_at = Column(DateTime, default=datetime.utcnow)

    space = relationship("Space", back_populates="placements")
    device = relationship("Device")

    def to_dict(self):
        return {
            "device_id": self.device.device_uuid if self.device else None,
            "device_name": self.device.device_name if self.device else None,
            "space_id": self.space_id,
            "x": self.x, "y": self.y,
            "rotation_deg": self.rotation_deg,
            "fov_deg": self.fov_deg,
            "range_m": self.range_m,
        }


class OrgMembership(Base):
    """Tracks which users belong to which organization accounts."""
    __tablename__ = "org_membership"

    id = Column(Integer, primary_key=True, autoincrement=True)
    org_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    member_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    status = Column(String(20), default="pending", index=True)  # pending | approved | declined
    invited_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    invite_code = Column(String(50), nullable=True)

    org = relationship("User", foreign_keys=[org_id], back_populates="org_memberships_as_org")
    member = relationship("User", foreign_keys=[member_id], back_populates="org_memberships_as_member")

    __table_args__ = (UniqueConstraint("org_id", "member_id", name="uq_org_member"),)


class InviteCode(Base):
    """Organization invite codes that users can use to request membership."""
    __tablename__ = "invite_code"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    org_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    max_uses = Column(Integer, default=100)
    uses_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

    org = relationship("User", foreign_keys=[org_id])


class Lab(Base):
    """A reproducible computational experiment in a research track.

    Labs are gated by the ``labs`` plan entitlement (Research), not by
    organization membership. Each lab ships a notebook template that the
    researcher completes with thothcraft-sdk and submits as ``.ipynb``.
    """
    __tablename__ = "lab"

    id = Column(Integer, primary_key=True, autoincrement=True)
    slug = Column(String(120), nullable=False, unique=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    track = Column(String(80), nullable=False, index=True)  # e.g. "dataset_engineering"
    level = Column(String(20), default="beginner")  # beginner | intermediate | advanced
    order_in_track = Column(Integer, default=0)
    objectives = Column(Text, nullable=True)  # JSON: [str]
    template_path = Column(String(500), nullable=True)  # notebook template in lab-templates/
    required_artifacts = Column(Text, nullable=True)  # JSON: ["notebook","metrics","figures",...]
    rubric = Column(Text, nullable=True)  # JSON: grading rubric for LabGrader
    max_score = Column(Integer, default=100)
    created_by = Column(Integer, ForeignKey("user_account.user_id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_published = Column(Boolean, default=True)

    creator = relationship("User", foreign_keys=[created_by])
    submissions = relationship("LabSubmission", back_populates="lab")


class LabSubmission(Base):
    """A researcher's notebook submission for a lab.

    ``status`` lifecycle: pending -> queued -> graded | failed.
    Notebook execution is deferred behind the LabGrader abstraction; no
    untrusted notebook code is executed by Brain.
    """
    __tablename__ = "lab_submission"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lab_id = Column(Integer, ForeignKey("lab.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    notebook_path = Column(String(500), nullable=False)  # stored .ipynb path
    notebook_metadata = Column(Text, nullable=True)  # JSON: extracted nb metadata
    status = Column(String(20), default="pending", index=True)  # pending|queued|graded|failed
    execution_status = Column(String(20), default="not_run")  # not_run|success|error
    score = Column(Float, nullable=True)
    max_score = Column(Integer, nullable=True)
    passed = Column(Boolean, nullable=True)
    feedback = Column(Text, nullable=True)  # JSON: [str] grader feedback
    artifacts = Column(Text, nullable=True)  # JSON: extracted artifact manifest
    submitted_at = Column(DateTime, default=datetime.utcnow)
    graded_at = Column(DateTime, nullable=True)

    lab = relationship("Lab", back_populates="submissions")
    user = relationship("User", foreign_keys=[user_id])

    __table_args__ = (UniqueConstraint("lab_id", "user_id", name="uq_lab_user"),)


class AuditEvent(Base):
    """Security-relevant audit trail.

    Recorded for pairing, unpairing, data deletion, downloads, model
    deployment and other sensitive actions. Append-only; never exposed
    to end users.
    """
    __tablename__ = "audit_event"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=True, index=True)
    device_id = Column(Integer, ForeignKey("device.device_id"), nullable=True, index=True)
    action = Column(String(80), nullable=False, index=True)  # e.g. "device.paired"
    detail = Column(Text, nullable=True)  # JSON context
    ip_address = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Payment(Base):
    """Payment records linked to Stripe events."""
    __tablename__ = "payment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    stripe_payment_intent = Column(String(255), nullable=True)
    stripe_invoice_id = Column(String(255), nullable=True)
    amount = Column(Integer, nullable=False)  # Amount in cents
    currency = Column(String(10), default="usd")
    plan = Column(String(50), nullable=True)  # which plan was purchased
    status = Column(String(50), default="pending")  # pending | succeeded | failed | refunded
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", foreign_keys=[user_id])


# ---------------------------------------------------------------------------
# Context model (Architecture §30–§35)
#
# Entities are canonical objects (persons, devices, spaces). Relationships
# are subject–predicate–object edges with validity windows. Evidence wraps
# observations/predictions with provenance — predictions are evidence,
# never truth. ContextState is a derived statement with evidence links;
# ContextEvent is the discrete transition emitted when a state changes.
# ---------------------------------------------------------------------------

class ContextEntity(Base):
    """A canonical context entity: person, device, space, or logical object."""
    __tablename__ = "context_entity"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    entity_key = Column(String(255), nullable=False, index=True)  # e.g. "person:gad"
    kind = Column(String(80), nullable=False, index=True)         # person|device|space|…
    name = Column(String(255), nullable=True)
    attributes = Column(Text, nullable=True)                      # JSON
    retired_at = Column(Float, nullable=True)                     # soft delete
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "entity_key", name="uq_context_entity_key"),
    )

    def to_dict(self):
        import json as _json
        return {
            "id": self.entity_key,
            "kind": self.kind,
            "name": self.name,
            "attributes": _json.loads(self.attributes) if self.attributes else {},
            "retired_at": self.retired_at,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
            "updated_at": self.updated_at.isoformat() + "Z" if self.updated_at else None,
        }


class ContextRelationship(Base):
    """Subject–predicate–object edge; valid_until=NULL means still valid."""
    __tablename__ = "context_relationship"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    subject = Column(String(255), nullable=False, index=True)
    predicate = Column(String(80), nullable=False, index=True)
    object = Column(String(255), nullable=False, index=True)
    valid_from = Column(Float, nullable=False)
    valid_until = Column(Float, nullable=True)
    confidence = Column(Float, default=1.0)
    source = Column(String(255), nullable=True)
    provenance = Column(Text, nullable=True)                      # JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        import json as _json
        return {
            "id": str(self.id),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "confidence": self.confidence,
            "source": self.source,
            "provenance": _json.loads(self.provenance) if self.provenance else {},
        }


class ContextEvidence(Base):
    """One piece of evidence feeding a ContextState (append-mostly)."""
    __tablename__ = "context_evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    external_id = Column(String(255), nullable=True)              # producer idempotency key
    evidence_key = Column(String(255), nullable=False, index=True)  # versioned key
    value = Column(Text, nullable=True)                           # JSON
    timestamp = Column(Float, nullable=False, index=True)
    source_id = Column(String(255), nullable=True, index=True)
    device_id = Column(String(255), nullable=True, index=True)
    prediction_id = Column(String(255), nullable=True)
    observation_id = Column(String(255), nullable=True)
    model_id = Column(String(255), nullable=True)
    model_version = Column(String(80), nullable=True)
    confidence = Column(Float, nullable=True)
    execution_class = Column(String(40), nullable=True)
    provenance = Column(Text, nullable=True)                      # JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "external_id",
                         name="uq_context_evidence_external"),
    )

    def to_dict(self):
        import json as _json
        return {
            "id": str(self.id),
            "external_id": self.external_id,
            "key": self.evidence_key,
            "value": _json.loads(self.value) if self.value else None,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "device_id": self.device_id,
            "prediction_id": self.prediction_id,
            "observation_id": self.observation_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "confidence": self.confidence,
            "execution_class": self.execution_class,
            "provenance": _json.loads(self.provenance) if self.provenance else {},
        }


class ContextState(Base):
    """A derived context statement with evidence links (current truth view)."""
    __tablename__ = "context_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    state_key = Column(String(255), nullable=False, index=True)     # versioned key
    # '' is the canonical global entity — a nullable column would break the
    # unique constraint below (Postgres treats NULLs as distinct).
    entity_id = Column(String(255), nullable=False, default="", server_default="", index=True)
    value = Column(Text, nullable=True)                           # JSON
    confidence = Column(Float, default=1.0)
    since = Column(Float, nullable=False)
    valid_until = Column(Float, nullable=True)
    evidence_ids = Column(Text, nullable=True)                    # JSON list
    estimator = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "state_key", "entity_id",
                         name="uq_context_state_key_entity"),
    )

    def to_dict(self):
        import json as _json
        return {
            "id": str(self.id),
            "key": self.state_key,
            "entity_id": self.entity_id or None,
            "value": _json.loads(self.value) if self.value else None,
            "confidence": self.confidence,
            "since": self.since,
            "valid_until": self.valid_until,
            "evidence_ids": _json.loads(self.evidence_ids) if self.evidence_ids else [],
            "estimator": self.estimator,
        }


class ContextEvent(Base):
    """Discrete transition emitted when a ContextState changes (append-only)."""
    __tablename__ = "context_event"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    event_key = Column(String(255), nullable=False, index=True)
    event_type = Column(String(20), nullable=False)               # entered|exited|changed
    entity_id = Column(String(255), nullable=True, index=True)
    state_id = Column(String(255), nullable=True)
    value = Column(Text, nullable=True)                           # JSON
    previous_value = Column(Text, nullable=True)                  # JSON
    confidence = Column(Float, nullable=True)
    timestamp = Column(Float, nullable=False, index=True)
    provenance = Column(Text, nullable=True)                      # JSON
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        import json as _json
        return {
            "id": str(self.id),
            "key": self.event_key,
            "event_type": self.event_type,
            "entity_id": self.entity_id,
            "state_id": self.state_id,
            "value": _json.loads(self.value) if self.value else None,
            "previous_value": _json.loads(self.previous_value) if self.previous_value else None,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "provenance": _json.loads(self.provenance) if self.provenance else {},
        }


class AutomationRule(Base):
    """A declarative automation rule evaluated by Brain's AutomationEngine.

    ``when``/``then`` are JSON payloads — no executable code is stored.
    """
    __tablename__ = "automation_rule"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    when = Column(Text, nullable=False)                           # JSON match spec
    then = Column(Text, nullable=False)                           # JSON action spec
    cooldown_s = Column(Float, default=0.0)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_automation_rule_name"),
    )

    def to_dict(self):
        import json as _json
        return {
            "id": str(self.id),
            "name": self.name,
            "when": _json.loads(self.when) if self.when else {},
            "then": _json.loads(self.then) if self.then else {},
            "cooldown_s": self.cooldown_s or 0.0,
            "enabled": bool(self.enabled),
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
            "updated_at": self.updated_at.isoformat() + "Z" if self.updated_at else None,
        }


class FaceBasis(Base):
    """A PCA eigenface basis — mean face + eigenvectors as .npz bytes.

    User-scoped: each user owns their basis (fitted from their enrolled
    photos and/or a public face dataset for stability) and their gallery.
    ``max_distance`` is the calibrated "very close" cutoff used by the
    edge recognizer for unknown rejection.
    """
    __tablename__ = "face_basis"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    name = Column(String(255), nullable=False, default="default")
    image_size = Column(Integer, nullable=False, default=64)
    n_components = Column(Integer, nullable=False, default=0)
    max_distance = Column(Float, default=0.0)
    data = Column(LargeBinary, nullable=False)                # .npz payload
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_face_basis_name"),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "image_size": self.image_size,
            "n_components": self.n_components,
            "max_distance": self.max_distance or 0.0,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }


class PersonAsset(Base):
    """One enrolled photo of a known person + its PCA projection.

    The assets DB for face recognition: ``photo`` keeps the enrolled
    image (audit/re-enrollment), ``projection`` is the eigenface weight
    vector the edge recognizer matches against. Multiple rows per
    (user, name) = multiple photos of the same person.
    """
    __tablename__ = "person_asset"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)                # person label
    basis_id = Column(Integer, ForeignKey("face_basis.id"), nullable=False)
    projection = Column(Text, nullable=False)                 # JSON weight vector
    photo = Column(LargeBinary, nullable=True)                # enrolled image
    photo_mime = Column(String(64), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self, include_projection: bool = True):
        out = {
            "id": str(self.id),
            "name": self.name,
            "basis_id": str(self.basis_id),
            "photo_mime": self.photo_mime or "",
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }
        if include_projection:
            try:
                out["projection"] = json.loads(self.projection) \
                    if self.projection else []
            except Exception:
                out["projection"] = []
        return out


# DO NOT run migrations or create tables at import time in serverless environments!
# Run this manually in a migration script or CLI, not here:
# Base.metadata.create_all(bind=engine)
