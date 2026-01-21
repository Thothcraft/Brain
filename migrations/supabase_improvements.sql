-- ============================================================================
-- SUPABASE DATABASE IMPROVEMENTS MIGRATION
-- ============================================================================
-- Run these commands in your Supabase SQL Editor (Dashboard > SQL Editor)
-- 
-- This migration includes:
-- 1. New columns for Supabase Storage integration
-- 2. Composite indexes for better query performance
-- 3. Realtime publication for training_job table
-- 4. Storage bucket creation
-- ============================================================================

-- ============================================================================
-- PART 1: SCHEMA CHANGES - New Columns for Storage
-- ============================================================================

-- Add storage_path column to file table (for Supabase Storage)
ALTER TABLE file 
ADD COLUMN IF NOT EXISTS storage_path VARCHAR(500) DEFAULT NULL;

COMMENT ON COLUMN file.storage_path IS 'Path in Supabase Storage (e.g., files/user_123/file_456/name.csv)';

-- Add storage_path column to trained_model table (for Supabase Storage)
ALTER TABLE trained_model 
ADD COLUMN IF NOT EXISTS storage_path VARCHAR(500) DEFAULT NULL;

COMMENT ON COLUMN trained_model.storage_path IS 'Path in Supabase Storage (e.g., models/user_123/model_456/name.pt)';


-- ============================================================================
-- PART 2: COMPOSITE INDEXES - Performance Optimization
-- ============================================================================

-- Training Job indexes
-- Index for listing jobs by user with status filter (most common query)
CREATE INDEX IF NOT EXISTS idx_training_job_user_status 
ON training_job(user_id, status);

-- Index for listing jobs by user ordered by creation date
CREATE INDEX IF NOT EXISTS idx_training_job_user_created 
ON training_job(user_id, created_at DESC);

-- Partial index for active jobs only (pending/running) - very efficient for monitoring
CREATE INDEX IF NOT EXISTS idx_training_job_active 
ON training_job(user_id, created_at DESC) 
WHERE status IN ('pending', 'running');

-- File indexes
-- Index for listing files by user ordered by upload date
CREATE INDEX IF NOT EXISTS idx_file_user_uploaded 
ON file(user_id, uploaded_at DESC);

-- Index for finding files by user and content type
CREATE INDEX IF NOT EXISTS idx_file_user_content_type 
ON file(user_id, content_type);

-- Index for files with storage_path (to find files in Supabase Storage)
CREATE INDEX IF NOT EXISTS idx_file_storage_path 
ON file(storage_path) 
WHERE storage_path IS NOT NULL;

-- Device File indexes
-- Index for device file sync queries
CREATE INDEX IF NOT EXISTS idx_device_file_user_device 
ON device_file(user_id, device_id, on_cloud);

-- Index for finding files that need upload
CREATE INDEX IF NOT EXISTS idx_device_file_upload_requested 
ON device_file(device_id, upload_requested) 
WHERE upload_requested = true;

-- Dataset indexes
-- Index for dataset files by dataset and label
CREATE INDEX IF NOT EXISTS idx_dataset_file_dataset_label 
ON dataset_file(dataset_id, label);

-- Trained Model indexes
-- Index for listing models by user
CREATE INDEX IF NOT EXISTS idx_trained_model_user_created 
ON trained_model(user_id, created_at DESC);

-- Index for finding models with storage_path
CREATE INDEX IF NOT EXISTS idx_trained_model_storage_path 
ON trained_model(storage_path) 
WHERE storage_path IS NOT NULL;

-- Preprocessing Pipeline indexes
CREATE INDEX IF NOT EXISTS idx_preprocessing_pipeline_user 
ON preprocessing_pipeline(user_id, created_at DESC);


-- ============================================================================
-- PART 3: ENABLE REALTIME FOR TRAINING JOBS
-- ============================================================================

-- Enable realtime for training_job table
-- This allows the frontend to subscribe to changes without polling

-- First, check if the publication exists and add the table
DO $$
BEGIN
    -- Try to add training_job to the realtime publication
    IF EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'supabase_realtime') THEN
        -- Check if table is already in publication
        IF NOT EXISTS (
            SELECT 1 FROM pg_publication_tables 
            WHERE pubname = 'supabase_realtime' 
            AND schemaname = 'public' 
            AND tablename = 'training_job'
        ) THEN
            ALTER PUBLICATION supabase_realtime ADD TABLE training_job;
            RAISE NOTICE 'Added training_job to supabase_realtime publication';
        ELSE
            RAISE NOTICE 'training_job already in supabase_realtime publication';
        END IF;
    ELSE
        RAISE NOTICE 'supabase_realtime publication does not exist';
    END IF;
END $$;

-- Alternative: If the above doesn't work, run this directly:
-- ALTER PUBLICATION supabase_realtime ADD TABLE training_job;


-- ============================================================================
-- PART 4: STORAGE BUCKET SETUP (Run in Supabase Dashboard)
-- ============================================================================

-- NOTE: Storage buckets must be created via Supabase Dashboard or API
-- Go to: Dashboard > Storage > New Bucket

-- Bucket 1: files
-- Name: files
-- Public: No (private)
-- File size limit: 200MB (209715200 bytes)
-- Allowed MIME types: Leave empty (allow all)

-- Bucket 2: models  
-- Name: models
-- Public: No (private)
-- File size limit: 200MB (209715200 bytes)
-- Allowed MIME types: Leave empty (allow all)

-- Storage Policies (run after creating buckets):

-- Policy: Users can upload their own files
CREATE POLICY "Users can upload own files" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
    bucket_id = 'files' AND
    (storage.foldername(name))[1] = 'user_' || auth.uid()::text
);

-- Policy: Users can read their own files
CREATE POLICY "Users can read own files" ON storage.objects
FOR SELECT TO authenticated
USING (
    bucket_id = 'files' AND
    (storage.foldername(name))[1] = 'user_' || auth.uid()::text
);

-- Policy: Users can delete their own files
CREATE POLICY "Users can delete own files" ON storage.objects
FOR DELETE TO authenticated
USING (
    bucket_id = 'files' AND
    (storage.foldername(name))[1] = 'user_' || auth.uid()::text
);

-- Policy: Users can upload their own models
CREATE POLICY "Users can upload own models" ON storage.objects
FOR INSERT TO authenticated
WITH CHECK (
    bucket_id = 'models' AND
    (storage.foldername(name))[1] = 'user_' || auth.uid()::text
);

-- Policy: Users can read their own models
CREATE POLICY "Users can read own models" ON storage.objects
FOR SELECT TO authenticated
USING (
    bucket_id = 'models' AND
    (storage.foldername(name))[1] = 'user_' || auth.uid()::text
);

-- Policy: Users can delete their own models
CREATE POLICY "Users can delete own models" ON storage.objects
FOR DELETE TO authenticated
USING (
    bucket_id = 'models' AND
    (storage.foldername(name))[1] = 'user_' || auth.uid()::text
);


-- ============================================================================
-- PART 5: VERIFY CHANGES
-- ============================================================================

-- Check indexes were created
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE schemaname = 'public' 
AND tablename IN ('training_job', 'file', 'device_file', 'dataset_file', 'trained_model', 'preprocessing_pipeline')
ORDER BY tablename, indexname;

-- Check realtime publication
SELECT * FROM pg_publication_tables WHERE pubname = 'supabase_realtime';

-- Check new columns exist
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'file' AND column_name = 'storage_path';

SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'trained_model' AND column_name = 'storage_path';


-- ============================================================================
-- ROLLBACK COMMANDS (if needed)
-- ============================================================================

-- To rollback indexes:
-- DROP INDEX IF EXISTS idx_training_job_user_status;
-- DROP INDEX IF EXISTS idx_training_job_user_created;
-- DROP INDEX IF EXISTS idx_training_job_active;
-- DROP INDEX IF EXISTS idx_file_user_uploaded;
-- DROP INDEX IF EXISTS idx_file_user_content_type;
-- DROP INDEX IF EXISTS idx_file_storage_path;
-- DROP INDEX IF EXISTS idx_device_file_user_device;
-- DROP INDEX IF EXISTS idx_device_file_upload_requested;
-- DROP INDEX IF EXISTS idx_dataset_file_dataset_label;
-- DROP INDEX IF EXISTS idx_trained_model_user_created;
-- DROP INDEX IF EXISTS idx_trained_model_storage_path;
-- DROP INDEX IF EXISTS idx_preprocessing_pipeline_user;

-- To rollback realtime:
-- ALTER PUBLICATION supabase_realtime DROP TABLE training_job;

-- To rollback columns:
-- ALTER TABLE file DROP COLUMN IF EXISTS storage_path;
-- ALTER TABLE trained_model DROP COLUMN IF EXISTS storage_path;
