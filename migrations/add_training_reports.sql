-- Migration: Add Training Reports and Data Validation Tables
-- This migration adds tables for:
-- 1. Training reports with shareable links
-- 2. File metadata validation
-- 3. FL session tracking

-- ============================================================================
-- TRAINING REPORTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS training_report (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(255) UNIQUE NOT NULL,
    job_id VARCHAR(255) REFERENCES training_job(job_id) ON DELETE SET NULL,
    user_id INTEGER NOT NULL REFERENCES user_account(user_id) ON DELETE CASCADE,
    
    -- Training info
    model_type VARCHAR(100),
    training_mode VARCHAR(50) DEFAULT 'central',  -- central, federated
    dataset_name VARCHAR(255),
    num_classes INTEGER DEFAULT 0,
    class_names TEXT,  -- JSON array
    
    -- Configuration
    epochs INTEGER DEFAULT 0,
    batch_size INTEGER DEFAULT 32,
    learning_rate FLOAT DEFAULT 0.001,
    optimizer VARCHAR(50) DEFAULT 'adam',
    
    -- Timing
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    total_time_seconds FLOAT DEFAULT 0,
    preprocessing_time_seconds FLOAT DEFAULT 0,
    training_time_seconds FLOAT DEFAULT 0,
    evaluation_time_seconds FLOAT DEFAULT 0,
    
    -- Metrics
    final_train_loss FLOAT,
    final_train_accuracy FLOAT,
    final_val_loss FLOAT,
    final_val_accuracy FLOAT,
    final_test_loss FLOAT,
    final_test_accuracy FLOAT,
    best_val_accuracy FLOAT DEFAULT 0,
    best_val_epoch INTEGER DEFAULT 0,
    
    -- Detailed metrics (JSON)
    epoch_metrics TEXT,  -- JSON array of per-epoch metrics
    class_metrics TEXT,  -- JSON array of per-class metrics
    confusion_matrix TEXT,  -- JSON object
    roc_curves TEXT,  -- JSON array
    pr_curves TEXT,  -- JSON array
    
    -- FL-specific
    fl_rounds TEXT,  -- JSON array of FL round metrics
    fl_algorithm VARCHAR(100),
    fl_num_clients INTEGER DEFAULT 0,
    
    -- Hardware
    device VARCHAR(50) DEFAULT 'cpu',
    gpu_name VARCHAR(255),
    peak_memory_mb FLOAT DEFAULT 0,
    
    -- Data stats
    train_samples INTEGER DEFAULT 0,
    val_samples INTEGER DEFAULT 0,
    test_samples INTEGER DEFAULT 0,
    input_shape TEXT,  -- JSON array
    
    -- Sharing
    share_token VARCHAR(64) UNIQUE,
    is_public BOOLEAN DEFAULT FALSE,
    
    -- Plots (base64 encoded)
    plots TEXT,  -- JSON object with plot names as keys
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for training_report
CREATE INDEX IF NOT EXISTS idx_training_report_user_id ON training_report(user_id);
CREATE INDEX IF NOT EXISTS idx_training_report_job_id ON training_report(job_id);
CREATE INDEX IF NOT EXISTS idx_training_report_share_token ON training_report(share_token);
CREATE INDEX IF NOT EXISTS idx_training_report_created_at ON training_report(created_at);

-- ============================================================================
-- FILE METADATA TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS file_metadata (
    id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES file(file_id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES user_account(user_id) ON DELETE CASCADE,
    
    -- File type and validation
    file_type VARCHAR(50) NOT NULL,  -- csi, general_csv, image, video, imu
    is_valid BOOLEAN DEFAULT FALSE,
    validation_errors TEXT,  -- JSON array
    validation_warnings TEXT,  -- JSON array
    
    -- Required metadata fields
    label VARCHAR(255),
    description TEXT,
    
    -- Type-specific metadata (JSON)
    type_metadata TEXT,
    
    -- Auto-extracted statistics
    statistics TEXT,  -- JSON object
    
    -- CSI-specific
    num_subcarriers INTEGER,
    sampling_rate_hz FLOAT,
    num_lines INTEGER,
    valid_lines INTEGER,
    bad_lines INTEGER,
    rssi_min FLOAT,
    rssi_max FLOAT,
    rssi_mean FLOAT,
    
    -- CSV-specific
    has_header BOOLEAN DEFAULT TRUE,
    delimiter VARCHAR(10) DEFAULT ',',
    num_columns INTEGER,
    column_names TEXT,  -- JSON array
    num_rows INTEGER,
    
    -- Image-specific
    width INTEGER,
    height INTEGER,
    channels INTEGER,
    image_format VARCHAR(50),
    color_mode VARCHAR(20),
    
    -- Video-specific
    fps FLOAT,
    duration_seconds FLOAT,
    num_frames INTEGER,
    codec VARCHAR(50),
    
    -- IMU-specific
    num_samples INTEGER,
    axes TEXT,  -- JSON array
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(file_id)
);

-- Indexes for file_metadata
CREATE INDEX IF NOT EXISTS idx_file_metadata_file_id ON file_metadata(file_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_user_id ON file_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_file_type ON file_metadata(file_type);
CREATE INDEX IF NOT EXISTS idx_file_metadata_is_valid ON file_metadata(is_valid);
CREATE INDEX IF NOT EXISTS idx_file_metadata_label ON file_metadata(label);

-- ============================================================================
-- FL SESSION TABLE (Enhanced)
-- ============================================================================

CREATE TABLE IF NOT EXISTS fl_session (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER NOT NULL REFERENCES user_account(user_id) ON DELETE CASCADE,
    
    -- Session info
    session_name VARCHAR(255) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,  -- fedavg, fedprox, etc.
    model_architecture VARCHAR(100),
    aggregation_method VARCHAR(100),
    client_selection VARCHAR(100),
    
    -- Configuration (JSON)
    server_config TEXT,
    client_config TEXT,
    algorithm_config TEXT,
    data_config TEXT,
    privacy_config TEXT,
    monitoring_config TEXT,
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending',  -- pending, running, completed, failed
    current_round INTEGER DEFAULT 0,
    total_rounds INTEGER DEFAULT 100,
    
    -- Metrics
    best_accuracy FLOAT DEFAULT 0,
    best_round INTEGER DEFAULT 0,
    round_metrics TEXT,  -- JSON array
    
    -- Privacy accounting
    privacy_budget_spent FLOAT DEFAULT 0,
    total_privacy_budget FLOAT DEFAULT 10,
    
    -- Timing
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Error handling
    error_message TEXT,
    
    -- Report link
    report_id VARCHAR(255) REFERENCES training_report(report_id) ON DELETE SET NULL
);

-- Indexes for fl_session
CREATE INDEX IF NOT EXISTS idx_fl_session_user_id ON fl_session(user_id);
CREATE INDEX IF NOT EXISTS idx_fl_session_status ON fl_session(status);
CREATE INDEX IF NOT EXISTS idx_fl_session_algorithm ON fl_session(algorithm);
CREATE INDEX IF NOT EXISTS idx_fl_session_created_at ON fl_session(created_at);

-- ============================================================================
-- FL CLIENT TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS fl_client (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL REFERENCES fl_session(session_id) ON DELETE CASCADE,
    device_id VARCHAR(255),
    
    -- Client info
    data_samples INTEGER DEFAULT 0,
    local_epochs_completed INTEGER DEFAULT 0,
    rounds_participated TEXT,  -- JSON array
    contribution_score FLOAT DEFAULT 0,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    compute_capability FLOAT DEFAULT 1.0,
    network_bandwidth FLOAT DEFAULT 1.0,
    
    -- Metrics history (JSON)
    metrics_history TEXT,
    
    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(client_id, session_id)
);

-- Indexes for fl_client
CREATE INDEX IF NOT EXISTS idx_fl_client_session_id ON fl_client(session_id);
CREATE INDEX IF NOT EXISTS idx_fl_client_device_id ON fl_client(device_id);
CREATE INDEX IF NOT EXISTS idx_fl_client_is_active ON fl_client(is_active);

-- ============================================================================
-- ADD REPORT LINK TO TRAINING JOB
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'training_job' AND column_name = 'report_id'
    ) THEN
        ALTER TABLE training_job ADD COLUMN report_id VARCHAR(255);
    END IF;
END $$;

-- ============================================================================
-- ADD METADATA VALIDATION STATUS TO FILE
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'file' AND column_name = 'is_validated'
    ) THEN
        ALTER TABLE file ADD COLUMN is_validated BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'file' AND column_name = 'validation_status'
    ) THEN
        ALTER TABLE file ADD COLUMN validation_status VARCHAR(50);
    END IF;
END $$;

-- Create index on validation status
CREATE INDEX IF NOT EXISTS idx_file_is_validated ON file(is_validated);
CREATE INDEX IF NOT EXISTS idx_file_validation_status ON file(validation_status);
