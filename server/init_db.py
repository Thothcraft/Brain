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

def initialize_database():
    """Initialize all required database tables."""
    logger.info("[INIT] Starting database initialization")
    
    try:
        ensure_trained_model_table()
        logger.info("[INIT] Database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"[INIT] Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    initialize_database()
