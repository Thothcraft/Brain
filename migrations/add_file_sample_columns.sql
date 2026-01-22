-- Migration: Add sample_content and data_type columns to file table
-- Purpose: Store preview data for quick access in preprocessing and training

-- Add sample_content column for storing first few lines of file
ALTER TABLE file ADD COLUMN IF NOT EXISTS sample_content TEXT;

-- Add data_type column for detected file type (csi, imu, sensor, etc.)
ALTER TABLE file ADD COLUMN IF NOT EXISTS data_type VARCHAR(50);

-- Add comment for documentation
COMMENT ON COLUMN file.sample_content IS 'First few lines of the file for quick preview (max ~10KB)';
COMMENT ON COLUMN file.data_type IS 'Detected data type: csi, imu, sensor, etc.';

-- Create index on data_type for filtering
CREATE INDEX IF NOT EXISTS idx_file_data_type ON file(data_type);
