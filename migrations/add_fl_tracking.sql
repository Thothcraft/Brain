-- Migration: Add FL Operation Tracking Tables
-- Created: 2026-01-24
-- Purpose: Comprehensive tracking for federated learning operations with per-round and per-client metrics

-- FL Sessions table - stores FL training session metadata
CREATE TABLE IF NOT EXISTS fl_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_name VARCHAR(255) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    model_architecture VARCHAR(50) NOT NULL,
    dataset VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    current_round INTEGER DEFAULT 0,
    total_rounds INTEGER NOT NULL,
    best_accuracy FLOAT DEFAULT 0.0,
    best_round INTEGER DEFAULT 0,
    privacy_budget_spent FLOAT DEFAULT 0.0,
    
    -- Configuration JSON
    server_config JSONB,
    client_config JSONB,
    algorithm_config JSONB,
    data_config JSONB,
    privacy_config JSONB,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    -- Error tracking
    error_message TEXT,
    
    -- Model storage
    global_model_path TEXT,
    
    -- User association
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL
);

-- FL Clients table - stores FL client/device information
CREATE TABLE IF NOT EXISTS fl_clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES fl_sessions(id) ON DELETE CASCADE,
    device_id VARCHAR(255) NOT NULL,
    
    -- Client type and connection
    is_remote BOOLEAN DEFAULT FALSE,
    remote_address VARCHAR(255),  -- IP:port for remote Thoth devices
    connection_status VARCHAR(20) DEFAULT 'connected',
    
    -- Client capabilities
    data_samples INTEGER DEFAULT 0,
    compute_capability FLOAT DEFAULT 1.0,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Aggregated metrics
    rounds_participated INTEGER[] DEFAULT '{}',
    rounds_failed INTEGER[] DEFAULT '{}',
    contribution_score FLOAT DEFAULT 0.0,
    total_training_time_ms FLOAT DEFAULT 0.0,
    total_communication_time_ms FLOAT DEFAULT 0.0,
    avg_accuracy FLOAT DEFAULT 0.0,
    best_accuracy FLOAT DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_update TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(session_id, device_id)
);

-- FL Round Metrics table - stores per-round aggregated metrics
CREATE TABLE IF NOT EXISTS fl_round_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES fl_sessions(id) ON DELETE CASCADE,
    round_num INTEGER NOT NULL,
    
    -- Global metrics
    global_loss FLOAT,
    global_accuracy FLOAT,
    train_loss FLOAT,
    
    -- Client statistics
    participating_clients INTEGER DEFAULT 0,
    avg_loss FLOAT,
    avg_accuracy FLOAT,
    min_accuracy FLOAT,
    max_accuracy FLOAT,
    std_accuracy FLOAT,
    
    -- Performance metrics
    aggregation_time_ms FLOAT DEFAULT 0.0,
    round_duration_ms FLOAT DEFAULT 0.0,
    communication_cost FLOAT DEFAULT 0.0,
    
    -- Convergence metrics
    convergence_rate FLOAT DEFAULT 0.0,
    fairness_index FLOAT DEFAULT 1.0,
    
    -- Client selection info
    selected_clients TEXT[],
    failed_clients TEXT[],
    
    -- Timestamps
    round_start_time TIMESTAMPTZ,
    round_end_time TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(session_id, round_num)
);

-- FL Client Round Metrics table - stores per-client per-round metrics
CREATE TABLE IF NOT EXISTS fl_client_round_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES fl_sessions(id) ON DELETE CASCADE,
    client_id UUID REFERENCES fl_clients(id) ON DELETE CASCADE,
    round_num INTEGER NOT NULL,
    
    -- Training metrics
    train_loss FLOAT,
    train_accuracy FLOAT,
    val_loss FLOAT,
    val_accuracy FLOAT,
    num_samples INTEGER,
    
    -- Performance metrics
    training_time_ms FLOAT DEFAULT 0.0,
    communication_time_ms FLOAT DEFAULT 0.0,
    model_size_bytes BIGINT DEFAULT 0,
    
    -- Status
    status VARCHAR(20) DEFAULT 'completed',  -- completed, failed, timeout
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(session_id, client_id, round_num)
);

-- FL Remote Devices table - stores Thoth devices available for FL participation
-- Reference: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html (SuperNode architecture)
CREATE TABLE IF NOT EXISTS fl_remote_devices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) NOT NULL UNIQUE,
    device_name VARCHAR(255),
    
    -- Connection info for Flower SuperNode
    -- Reference: https://flower.ai/docs/framework/how-to-run-flower-using-docker.html
    ip_address VARCHAR(45) NOT NULL,  -- IPv4 or IPv6
    port INTEGER DEFAULT 9094,  -- Default SuperNode ClientAppIO port
    
    -- Device capabilities
    compute_capability FLOAT DEFAULT 1.0,
    available_memory_mb INTEGER,
    cpu_cores INTEGER,
    has_gpu BOOLEAN DEFAULT FALSE,
    gpu_memory_mb INTEGER,
    
    -- Status
    status VARCHAR(20) DEFAULT 'offline',  -- online, offline, busy, error
    last_heartbeat TIMESTAMPTZ,
    
    -- Data info
    available_datasets TEXT[],  -- Datasets available on this device
    data_samples_available INTEGER DEFAULT 0,
    
    -- FL participation history
    sessions_participated INTEGER DEFAULT 0,
    total_rounds_completed INTEGER DEFAULT 0,
    avg_contribution_score FLOAT DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- User association
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_fl_sessions_status ON fl_sessions(status);
CREATE INDEX IF NOT EXISTS idx_fl_sessions_user ON fl_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_fl_sessions_created ON fl_sessions(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_fl_clients_session ON fl_clients(session_id);
CREATE INDEX IF NOT EXISTS idx_fl_clients_device ON fl_clients(device_id);
CREATE INDEX IF NOT EXISTS idx_fl_clients_active ON fl_clients(is_active);

CREATE INDEX IF NOT EXISTS idx_fl_round_metrics_session ON fl_round_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_fl_round_metrics_round ON fl_round_metrics(session_id, round_num);

CREATE INDEX IF NOT EXISTS idx_fl_client_round_metrics_session ON fl_client_round_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_fl_client_round_metrics_client ON fl_client_round_metrics(client_id);
CREATE INDEX IF NOT EXISTS idx_fl_client_round_metrics_round ON fl_client_round_metrics(session_id, round_num);

CREATE INDEX IF NOT EXISTS idx_fl_remote_devices_status ON fl_remote_devices(status);
CREATE INDEX IF NOT EXISTS idx_fl_remote_devices_user ON fl_remote_devices(user_id);

-- Enable RLS
ALTER TABLE fl_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE fl_clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE fl_round_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE fl_client_round_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE fl_remote_devices ENABLE ROW LEVEL SECURITY;

-- RLS Policies for fl_sessions
CREATE POLICY "Users can view their own FL sessions"
    ON fl_sessions FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can create FL sessions"
    ON fl_sessions FOR INSERT
    WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can update their own FL sessions"
    ON fl_sessions FOR UPDATE
    USING (auth.uid() = user_id OR user_id IS NULL);

-- RLS Policies for fl_clients (inherit from session)
CREATE POLICY "Users can view clients in their sessions"
    ON fl_clients FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM fl_sessions s 
        WHERE s.id = fl_clients.session_id 
        AND (s.user_id = auth.uid() OR s.user_id IS NULL)
    ));

CREATE POLICY "Users can manage clients in their sessions"
    ON fl_clients FOR ALL
    USING (EXISTS (
        SELECT 1 FROM fl_sessions s 
        WHERE s.id = fl_clients.session_id 
        AND (s.user_id = auth.uid() OR s.user_id IS NULL)
    ));

-- RLS Policies for fl_round_metrics
CREATE POLICY "Users can view metrics for their sessions"
    ON fl_round_metrics FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM fl_sessions s 
        WHERE s.id = fl_round_metrics.session_id 
        AND (s.user_id = auth.uid() OR s.user_id IS NULL)
    ));

CREATE POLICY "Users can insert metrics for their sessions"
    ON fl_round_metrics FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM fl_sessions s 
        WHERE s.id = fl_round_metrics.session_id 
        AND (s.user_id = auth.uid() OR s.user_id IS NULL)
    ));

-- RLS Policies for fl_client_round_metrics
CREATE POLICY "Users can view client metrics for their sessions"
    ON fl_client_round_metrics FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM fl_sessions s 
        WHERE s.id = fl_client_round_metrics.session_id 
        AND (s.user_id = auth.uid() OR s.user_id IS NULL)
    ));

CREATE POLICY "Users can insert client metrics for their sessions"
    ON fl_client_round_metrics FOR INSERT
    WITH CHECK (EXISTS (
        SELECT 1 FROM fl_sessions s 
        WHERE s.id = fl_client_round_metrics.session_id 
        AND (s.user_id = auth.uid() OR s.user_id IS NULL)
    ));

-- RLS Policies for fl_remote_devices
CREATE POLICY "Users can view their own remote devices"
    ON fl_remote_devices FOR SELECT
    USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can manage their own remote devices"
    ON fl_remote_devices FOR ALL
    USING (auth.uid() = user_id OR user_id IS NULL);

-- Function to update device heartbeat
CREATE OR REPLACE FUNCTION update_fl_device_heartbeat(p_device_id VARCHAR)
RETURNS void AS $$
BEGIN
    UPDATE fl_remote_devices 
    SET last_heartbeat = NOW(), 
        status = 'online',
        updated_at = NOW()
    WHERE device_id = p_device_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get available FL devices for a session
CREATE OR REPLACE FUNCTION get_available_fl_devices(p_min_capability FLOAT DEFAULT 0.0)
RETURNS TABLE (
    device_id VARCHAR,
    device_name VARCHAR,
    ip_address VARCHAR,
    port INTEGER,
    compute_capability FLOAT,
    data_samples_available INTEGER,
    status VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.device_id,
        d.device_name,
        d.ip_address,
        d.port,
        d.compute_capability,
        d.data_samples_available,
        d.status
    FROM fl_remote_devices d
    WHERE d.status = 'online'
      AND d.compute_capability >= p_min_capability
      AND d.last_heartbeat > NOW() - INTERVAL '5 minutes'
    ORDER BY d.compute_capability DESC, d.data_samples_available DESC;
END;
$$ LANGUAGE plpgsql;
