-- Create tables for cloud training functionality
-- Run this on your PostgreSQL database

-- 1. Training Dataset table
CREATE TABLE IF NOT EXISTS training_dataset (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. Dataset File table (links files to datasets with labels)
-- Note: No unique constraint on (dataset_id, file_id) to allow multiple labels
CREATE TABLE IF NOT EXISTS dataset_file (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES training_dataset(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES file(file_id),
    label VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. Training Job table
CREATE TABLE IF NOT EXISTS training_job (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
    dataset_id INTEGER REFERENCES training_dataset(id),
    model_type VARCHAR(50) NOT NULL,
    training_mode VARCHAR(50) NOT NULL,
    config TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER,
    metrics TEXT,
    best_metrics TEXT,
    model_path VARCHAR(500),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- 4. Trained Model table
CREATE TABLE IF NOT EXISTS trained_model (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
    job_id VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    architecture VARCHAR(50),
    accuracy INTEGER,
    size_bytes BIGINT,
    model_data BYTEA,
    config TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_training_dataset_user_id ON training_dataset(user_id);
CREATE INDEX IF NOT EXISTS idx_dataset_file_dataset_id ON dataset_file(dataset_id);
CREATE INDEX IF NOT EXISTS idx_dataset_file_file_id ON dataset_file(file_id);
CREATE INDEX IF NOT EXISTS idx_training_job_user_id ON training_job(user_id);
CREATE INDEX IF NOT EXISTS idx_training_job_status ON training_job(status);
CREATE INDEX IF NOT EXISTS idx_trained_model_user_id ON trained_model(user_id);

-- Grant permissions (adjust username as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO lms_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO lms_user;
