-- Migration: Add folder table and folder_id to file table
-- Date: 2025-01-24

-- Create folder table
CREATE TABLE IF NOT EXISTS folder (
    folder_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    user_id INTEGER NOT NULL REFERENCES user_account(user_id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES folder(folder_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- Create indexes for folder table
CREATE INDEX IF NOT EXISTS idx_folder_user_id ON folder(user_id);
CREATE INDEX IF NOT EXISTS idx_folder_parent_id ON folder(parent_id);

-- Add folder_id column to file table if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'file' AND column_name = 'folder_id'
    ) THEN
        ALTER TABLE file ADD COLUMN folder_id INTEGER REFERENCES folder(folder_id) ON DELETE SET NULL;
        CREATE INDEX idx_file_folder_id ON file(folder_id);
    END IF;
END $$;

-- Add labels column to file table if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'file' AND column_name = 'labels'
    ) THEN
        ALTER TABLE file ADD COLUMN labels TEXT;
    END IF;
END $$;

-- Create trigger to update updated_at on folder changes
CREATE OR REPLACE FUNCTION update_folder_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS folder_updated_at_trigger ON folder;
CREATE TRIGGER folder_updated_at_trigger
    BEFORE UPDATE ON folder
    FOR EACH ROW
    EXECUTE FUNCTION update_folder_updated_at();
