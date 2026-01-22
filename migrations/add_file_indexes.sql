-- Migration: Add indexes to file table for query performance
-- This fixes the query timeout on activity_endpoints.py

-- Index on user_id for filtering files by user
CREATE INDEX IF NOT EXISTS idx_file_user_id ON file(user_id);

-- Index on uploaded_at for time-based filtering and sorting
CREATE INDEX IF NOT EXISTS idx_file_uploaded_at ON file(uploaded_at DESC);

-- Composite index for the common query pattern: filter by user_id and uploaded_at, order by uploaded_at
CREATE INDEX IF NOT EXISTS idx_file_user_uploaded ON file(user_id, uploaded_at DESC);

-- Also add indexes for other tables used in activity queries

-- TrainingJob indexes
CREATE INDEX IF NOT EXISTS idx_training_job_user_id ON training_job(user_id);
CREATE INDEX IF NOT EXISTS idx_training_job_created_at ON training_job(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_job_completed_at ON training_job(completed_at DESC);

-- Device indexes (if not already present)
CREATE INDEX IF NOT EXISTS idx_device_user_id ON device(user_id);
CREATE INDEX IF NOT EXISTS idx_device_last_seen ON device(last_seen DESC);

-- Query table indexes
CREATE INDEX IF NOT EXISTS idx_query_user_id ON query(user_id);
CREATE INDEX IF NOT EXISTS idx_query_created_at ON query(created_at DESC);
