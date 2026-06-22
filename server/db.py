"""Database Models Module for LMS Platform.

This module defines the database models and connection setup for the LMS platform.
It includes the User model and database connection configuration.
"""

import os
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

print(f"[DB] Using DATABASE_URL: {DATABASE_URL}")

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
        # dashboard polls through multiple services, so 30 seconds was too
        # aggressive and made active Thoth devices appear offline.
        if self.last_seen:
            age = (datetime.utcnow() - self.last_seen).total_seconds()
            is_online = self.online and age <= 180
        else:
            is_online = False

        return {
            "device_id": self.device_uuid,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "online": is_online,
            "battery_level": self.battery_level,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "ip_address": self.ip_address,
            "mac_address": self.mac_address,
            "device_uuid": self.device_uuid,
            "user_id": self.userId,
            "hardware_info": hw_info,
            "portal_upload_allowed": bool(hw_info.get("portal_upload_allowed", True)) if isinstance(hw_info, dict) else True,
            "deployment_requests_allowed": bool(hw_info.get("deployment_requests_allowed", True)) if isinstance(hw_info, dict) else True,
            "cloud_sync_allowed": bool(hw_info.get("cloud_sync_allowed", True)) if isinstance(hw_info, dict) else True,
            "approved": self.approved if self.approved is not None else False
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
    preprocessing_pipeline_id = Column(Integer, ForeignKey("preprocessing_pipeline.id"), nullable=True)
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
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True, index=True)
    
    # Relationships
    user = relationship("User")
    dataset = relationship("TrainingDataset", foreign_keys=[dataset_id])
    test_dataset = relationship("TrainingDataset", foreign_keys=[test_dataset_id])
    preprocessing_pipeline = relationship("PreprocessingPipeline")
    
    def to_dict(self):
        import json
        return {
            "id": self.id,
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset.name if self.dataset else None,
            "test_dataset_id": self.test_dataset_id,
            "preprocessing_pipeline_id": self.preprocessing_pipeline_id,
            "preprocessing_pipeline_name": self.preprocessing_pipeline.name if self.preprocessing_pipeline else None,
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


class PreprocessingPipeline(Base):
    """Stores preprocessing pipelines for CSI/IMU data."""
    __tablename__ = "preprocessing_pipeline"
    
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    data_type = Column(String(50), nullable=False, default="csi")  # csi, imu, sensor
    
    # Pipeline configuration as JSON
    config = Column(Text, nullable=False)  # JSON: blocks, connections, parameters
    
    # Output configuration
    output_shape = Column(String(50), default="flattened")  # flattened, sequence
    include_phase = Column(Boolean, default=True)
    window_size = Column(Integer, default=1000)
    
    # Subcarrier filtering
    filter_subcarriers = Column(Boolean, default=True)
    subcarrier_start = Column(Integer, default=5)
    subcarrier_end = Column(Integer, default=32)
    
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    
    def to_dict(self):
        import json
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type,
            "config": json.loads(self.config) if self.config else {},
            "output_shape": self.output_shape,
            "include_phase": self.include_phase,
            "window_size": self.window_size,
            "filter_subcarriers": self.filter_subcarriers,
            "subcarrier_start": self.subcarrier_start,
            "subcarrier_end": self.subcarrier_end,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
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
    """Practice labs created by admins, visible to approved org members."""
    __tablename__ = "lab"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    sensor_type = Column(String(50), nullable=False)  # camera | wifi_sensing | cwmf
    difficulty = Column(String(20), default="beginner")  # beginner | intermediate | advanced
    questions = Column(Text, nullable=False)  # JSON: [{id, type, prompt, options?, correct_answer}]
    max_score = Column(Integer, default=100)
    created_by = Column(Integer, ForeignKey("user_account.user_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_published = Column(Boolean, default=True)

    creator = relationship("User", foreign_keys=[created_by])
    submissions = relationship("LabSubmission", back_populates="lab")


class LabSubmission(Base):
    """A member's submission for a lab."""
    __tablename__ = "lab_submission"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lab_id = Column(Integer, ForeignKey("lab.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    org_id = Column(Integer, ForeignKey("user_account.user_id"), nullable=False, index=True)
    answers = Column(Text, nullable=False)  # JSON: {question_id: answer}
    score = Column(Float, nullable=True)
    max_score = Column(Integer, nullable=True)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    graded_at = Column(DateTime, nullable=True)
    feedback = Column(Text, nullable=True)

    lab = relationship("Lab", back_populates="submissions")
    user = relationship("User", foreign_keys=[user_id])

    __table_args__ = (UniqueConstraint("lab_id", "user_id", name="uq_lab_user"),)


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


# DO NOT run migrations or create tables at import time in serverless environments!
# Run this manually in a migration script or CLI, not here:
# Base.metadata.create_all(bind=engine)
