-- Migration: Add indexes for activity endpoints performance
-- This significantly speeds up the /activity/recent and /activity/stats endpoints
-- Run this in your PostgreSQL database (Supabase SQL Editor)

-- Index for Device table - speeds up device activity queries
CREATE INDEX IF NOT EXISTS idx_device_userid_lastseen 
ON device (user_id, last_seen DESC);

CREATE INDEX IF NOT EXISTS idx_device_userid_online 
ON device (user_id, online);

-- Index for File table - speeds up file upload queries
CREATE INDEX IF NOT EXISTS idx_file_userid_uploadedat 
ON file (user_id, uploaded_at DESC);

-- Index for TrainingJob table - speeds up training job queries
CREATE INDEX IF NOT EXISTS idx_trainingjob_userid_createdat 
ON training_job (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_trainingjob_userid_status 
ON training_job (user_id, status);

CREATE INDEX IF NOT EXISTS idx_trainingjob_userid_completedat 
ON training_job (user_id, completed_at DESC);

-- Index for TrainedModel table - speeds up model queries
CREATE INDEX IF NOT EXISTS idx_trainedmodel_userid_createdat 
ON trained_model (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_trainedmodel_userid_accuracy 
ON trained_model (user_id, accuracy DESC);

-- Index for Query table (AI queries) - speeds up query history
CREATE INDEX IF NOT EXISTS idx_query_userid_createdat 
ON query (user_id, created_at DESC);

-- Analyze tables to update statistics for query planner
ANALYZE device;
ANALYZE file;
ANALYZE training_job;
ANALYZE trained_model;
ANALYZE query;
