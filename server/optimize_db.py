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
            try:
                db.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_training_datasets_user_id 
                    ON training_dataset(user_id DESC)
                """))
            except Exception as e:
                logger.warning(f"[OPTIMIZE] Could not create training_datasets_user_id index: {e}")
            
            # Index for user_id queries on training jobs
            try:
                db.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_training_jobs_user_id 
                    ON training_job(user_id DESC, created_at DESC)
                """))
            except Exception as e:
                logger.warning(f"[OPTIMIZE] Could not create training_jobs_user_id index: {e}")
            
            # Index for status queries on training jobs
            try:
                db.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_training_jobs_status 
                    ON training_job(status, created_at DESC)
                """))
            except Exception as e:
                logger.warning(f"[OPTIMIZE] Could not create training_jobs_status index: {e}")
            
            # Index for user_id queries on trained models
            try:
                db.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_trained_models_user_id 
                    ON trained_model(user_id DESC, created_at DESC)
                """))
            except Exception as e:
                logger.warning(f"[OPTIMIZE] Could not create trained_models_user_id index: {e}")
            
            # Index for user_id queries on files
            try:
                db.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_files_user_id 
                    ON file(user_id DESC, uploaded_at DESC)
                """))
            except Exception as e:
                logger.warning(f"[OPTIMIZE] Could not create files_user_id index: {e}")
            
            # Composite index for training jobs (user_id + status)
            try:
                db.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_training_jobs_user_status 
                    ON training_job(user_id, status, created_at DESC)
                """))
            except Exception as e:
                logger.warning(f"[OPTIMIZE] Could not create training_jobs_user_status index: {e}")
            
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
            
            # Update statistics for all relevant tables (using proper table names)
            tables = [
                'training_dataset',
                'training_job', 
                'trained_model',
                'file',
                'user_account'  # Correct table name from db.py
            ]
            
            for table in tables:
                try:
                    db.execute(text(f"ANALYZE {table}"))
                except Exception as e:
                    logger.warning(f"[OPTIMIZE] Could not analyze table {table}: {e}")
            
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
