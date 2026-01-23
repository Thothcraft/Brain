-- SQL Commands to Clear File Records for Fresh Start
-- Run these commands in your PostgreSQL database (Supabase SQL Editor)
-- This allows users to re-upload files and fill out metadata manually

-- ============================================================
-- OPTION 1: Delete ALL file records (complete reset)
-- ============================================================

-- Delete all files from the file table
DELETE FROM file;

-- Reset the file ID sequence (optional, starts IDs from 1 again)
-- ALTER SEQUENCE file_file_id_seq RESTART WITH 1;


-- ============================================================
-- OPTION 2: Delete files for a SPECIFIC USER
-- ============================================================

-- Replace 1117 with the actual user ID
-- DELETE FROM file WHERE user_id = 1117;


-- ============================================================
-- OPTION 3: Delete files uploaded before a certain date
-- ============================================================

-- Delete files uploaded more than 30 days ago
-- DELETE FROM file WHERE uploaded_at < NOW() - INTERVAL '30 days';


-- ============================================================
-- OPTION 4: Delete files that don't follow the naming convention
-- ============================================================

-- Delete files that don't start with valid prefixes
-- DELETE FROM file 
-- WHERE file_name NOT LIKE 'csi_%' 
--   AND file_name NOT LIKE 'imu_%' 
--   AND file_name NOT LIKE 'img_%' 
--   AND file_name NOT LIKE 'vid_%'
--   AND file_name NOT LIKE 'audio_%'
--   AND file_name NOT LIKE 'sensor_%';


-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check how many files exist
SELECT COUNT(*) as total_files FROM file;

-- Check files per user
SELECT user_id, COUNT(*) as file_count 
FROM file 
GROUP BY user_id 
ORDER BY file_count DESC;

-- Check file types distribution
SELECT 
    CASE 
        WHEN file_name LIKE 'csi_%' THEN 'CSI'
        WHEN file_name LIKE 'imu_%' THEN 'IMU'
        WHEN file_name LIKE 'img_%' THEN 'Image'
        WHEN file_name LIKE 'vid_%' THEN 'Video'
        WHEN file_name LIKE 'audio_%' THEN 'Audio'
        WHEN file_name LIKE 'sensor_%' THEN 'Sensor'
        ELSE 'Other'
    END as file_type,
    COUNT(*) as count
FROM file
GROUP BY file_type
ORDER BY count DESC;


-- ============================================================
-- ALSO CLEAR RELATED TABLES (if needed)
-- ============================================================

-- Clear datasets (which may reference files)
-- DELETE FROM dataset;

-- Clear training jobs
-- DELETE FROM training_job;

-- Clear trained models
-- DELETE FROM trained_model;
