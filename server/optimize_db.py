"""Database optimization script for Supabase Pro Plan.

This script adds indexes and optimizes the database for better performance.
"""

import logging
from sqlalchemy import text, create_engine
from .db import engine, SessionLocal, DATABASE_URL

logger = logging.getLogger(__name__)

def create_performance_indexes():
    """Create indexes to improve query performance."""
    try:
        with SessionLocal() as db:
            logger.info("[OPTIMIZE] Creating performance indexes...")
            
            # Index for user_id queries on datasets
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_datasets_user_id 
                ON training_dataset(user_id DESC)
            """))
            
            # Index for user_id queries on training jobs
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id 
                ON training_job(user_id DESC, created_at DESC)
            """))
            
            # Index for status queries on training jobs
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_jobs_status 
                ON training_job(status, created_at DESC)
            """))
            
            # Index for user_id queries on trained models
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_trained_models_user_id 
                ON trained_model(user_id DESC, created_at DESC)
            """))
            
            # Index for user_id queries on files
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_files_user_id 
                ON file(userId DESC, uploaded_at DESC)
            """))
            
            # Composite index for training jobs (user_id + status)
            db.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_training_jobs_user_status 
                ON training_job(user_id, status, created_at DESC)
            """))
            
            db.commit()
            logger.info("[OPTIMIZE] Performance indexes created successfully")
            return True
            
    except Exception as e:
        logger.error(f"[OPTIMIZE] Error creating indexes: {e}")
        return False

def analyze_table_statistics():
    """Update table statistics for better query planning."""
    try:
        with SessionLocal() as db:
            logger.info("[OPTIMIZE] Updating table statistics...")
            
            # Update statistics for all relevant tables
            tables = [
                'training_dataset',
                'training_job', 
                'trained_model',
                'file',
                'user'
            ]
            
            for table in tables:
                db.execute(text(f"ANALYZE {table}"))
            
            db.commit()
            logger.info("[OPTIMIZE] Table statistics updated successfully")
            return True
            
    except Exception as e:
        logger.error(f"[OPTIMIZE] Error updating statistics: {e}")
        return False

def optimize_database():
    """Run all database optimizations."""
    logger.info("[OPTIMIZE] Starting database optimization...")
    
    success = True
    
    # Create performance indexes
    if not create_performance_indexes():
        success = False
    
    # Update table statistics
    if not analyze_table_statistics():
        success = False
    
    if success:
        logger.info("[OPTIMIZE] Database optimization completed successfully")
    else:
        logger.error("[OPTIMIZE] Database optimization completed with errors")
    
    return success

if __name__ == "__main__":
    optimize_database()
