"""Database initialization script.

This script ensures all required database tables exist.
Run this once during application startup.
"""

import logging
from sqlalchemy import text, create_engine
from .db import engine, SessionLocal, DATABASE_URL

logger = logging.getLogger(__name__)

def ensure_trained_model_table():
    """Ensure the trained_model table exists."""
    try:
        # Use a direct connection with minimal settings for table creation
        direct_engine = create_engine(
            DATABASE_URL,
            pool_size=1,
            max_overflow=0,
            pool_timeout=10,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "sslmode": "require",
                "options": "-c statement_timeout=10000"
            }
        )
        
        with direct_engine.connect() as conn:
            # Check if table exists first
            table_check = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'trained_model'
                )
            """)).scalar()
            
            if not table_check:
                logger.info("[INIT] Creating trained_model table")
                conn.execute(text("""
                    CREATE TABLE trained_model (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        job_id VARCHAR(255),
                        name VARCHAR(255) NOT NULL,
                        architecture VARCHAR(50),
                        accuracy FLOAT,
                        size_bytes BIGINT,
                        model_data BYTEA,
                        config TEXT,
                        is_pinned BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
                logger.info("[INIT] Table created successfully")
            else:
                logger.info("[INIT] trained_model table already exists")
                
    except Exception as e:
        logger.error(f"[INIT] Error creating trained_model table: {e}")
        # Don't raise the exception - allow the application to continue
        return False
    
    return True

def ensure_approved_column():
    """Ensure the approved column exists on the device table."""
    try:
        direct_engine = create_engine(
            DATABASE_URL,
            pool_size=1,
            max_overflow=0,
            pool_timeout=10,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "sslmode": "require",
                "options": "-c statement_timeout=10000"
            }
        )
        with direct_engine.connect() as conn:
            col_check = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_name = 'device' AND column_name = 'approved'
                )
            """)).scalar()
            if not col_check:
                logger.info("[INIT] Adding 'approved' column to device table")
                conn.execute(text("""
                    ALTER TABLE device ADD COLUMN approved BOOLEAN NOT NULL DEFAULT FALSE
                """))
                conn.commit()
                logger.info("[INIT] 'approved' column added successfully")
            else:
                logger.info("[INIT] 'approved' column already exists")
    except Exception as e:
        logger.error(f"[INIT] Error ensuring approved column: {e}")
        return False
    return True


def ensure_product_core_schema():
    """Apply small, idempotent schema improvements required by the product UI."""
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                ALTER TABLE user_account
                    ADD COLUMN IF NOT EXISTS email VARCHAR(320),
                    ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS supabase_auth_user_id VARCHAR(36)
            """))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_user_account_email ON user_account (lower(email)) WHERE email IS NOT NULL"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_user_account_supabase_auth ON user_account (supabase_auth_user_id) WHERE supabase_auth_user_id IS NOT NULL"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_payment_invoice ON payment (stripe_invoice_id) WHERE stripe_invoice_id IS NOT NULL"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_user_activity ON device (user_id, approved, last_seen DESC)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_file_visible_minutes ON device_file (device_id, modified_at DESC) WHERE on_device = TRUE"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_file_cloud_history ON device_file (user_id, modified_at DESC) WHERE on_cloud = TRUE"))
        return True
    except Exception as e:
        logger.error(f"[INIT] Error ensuring product core schema: {e}")
        return False


def ensure_device_deployment_table():
    """Ensure the device_deployment table exists for pull-based model delivery."""
    try:
        direct_engine = create_engine(
            DATABASE_URL,
            pool_size=1,
            max_overflow=0,
            pool_timeout=10,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "sslmode": "require",
                "options": "-c statement_timeout=10000"
            }
        )
        with direct_engine.connect() as conn:
            table_check = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'device_deployment'
                )
            """)).scalar()
            if not table_check:
                logger.info("[INIT] Creating device_deployment table")
                conn.execute(text("""
                    CREATE TABLE device_deployment (
                        id SERIAL PRIMARY KEY,
                        deployment_id VARCHAR(255) UNIQUE NOT NULL,
                        device_uuid VARCHAR(255) NOT NULL,
                        model_id INTEGER NOT NULL,
                        user_id INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        status VARCHAR(50) NOT NULL DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        delivered_at TIMESTAMP
                    )
                """))
                conn.execute(text("CREATE INDEX idx_device_deployment_device ON device_deployment(device_uuid)"))
                conn.execute(text("CREATE INDEX idx_device_deployment_status ON device_deployment(status)"))
                conn.commit()
                logger.info("[INIT] device_deployment table created")
            else:
                logger.info("[INIT] device_deployment table already exists")
    except Exception as e:
        logger.error(f"[INIT] Error creating device_deployment table: {e}")
        return False
    return True


def initialize_database():
    """Initialize all required database tables."""
    logger.info("[INIT] Starting database initialization")
    
    try:
        results = {
            "product_core": ensure_product_core_schema(),
            "trained_model": ensure_trained_model_table(),
            "device.approved": ensure_approved_column(),
            "device_deployment": ensure_device_deployment_table(),
        }
        failed = [name for name, succeeded in results.items() if not succeeded]
        if failed:
            logger.warning("[INIT] Database initialization incomplete; failed checks: %s", ", ".join(failed))
            return False
        logger.info("[INIT] Database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"[INIT] Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    initialize_database()
