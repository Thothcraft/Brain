"""Database initialization script.

This script ensures all required database tables exist.
Run this once during application startup.
"""

import logging
from sqlalchemy import text, create_engine
from .db import engine, SessionLocal, DATABASE_URL

logger = logging.getLogger(__name__)

def ensure_trained_model_table():
    """Ensure the trained_model table exists."""
    try:
        # Use a direct connection with minimal settings for table creation
        direct_engine = create_engine(
            DATABASE_URL,
            pool_size=1,
            max_overflow=0,
            pool_timeout=10,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "sslmode": "require",
                "options": "-c statement_timeout=10000"
            }
        )
        
        with direct_engine.connect() as conn:
            # Check if table exists first
            table_check = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'trained_model'
                )
            """)).scalar()
            
            if not table_check:
                logger.info("[INIT] Creating trained_model table")
                conn.execute(text("""
                    CREATE TABLE trained_model (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        job_id VARCHAR(255),
                        name VARCHAR(255) NOT NULL,
                        architecture VARCHAR(50),
                        accuracy FLOAT,
                        size_bytes BIGINT,
                        model_data BYTEA,
                        config TEXT,
                        is_pinned BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
                logger.info("[INIT] Table created successfully")
            else:
                logger.info("[INIT] trained_model table already exists")
                
    except Exception as e:
        logger.error(f"[INIT] Error creating trained_model table: {e}")
        # Don't raise the exception - allow the application to continue
        return False
    
    return True

def ensure_approved_column():
    """Ensure the approved column exists on the device table."""
    try:
        direct_engine = create_engine(
            DATABASE_URL,
            pool_size=1,
            max_overflow=0,
            pool_timeout=10,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "sslmode": "require",
                "options": "-c statement_timeout=10000"
            }
        )
        with direct_engine.connect() as conn:
            col_check = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns
                    WHERE table_name = 'device' AND column_name = 'approved'
                )
            """)).scalar()
            if not col_check:
                logger.info("[INIT] Adding 'approved' column to device table")
                conn.execute(text("""
                    ALTER TABLE device ADD COLUMN approved BOOLEAN NOT NULL DEFAULT FALSE
                """))
                conn.commit()
                logger.info("[INIT] 'approved' column added successfully")
            else:
                logger.info("[INIT] 'approved' column already exists")
    except Exception as e:
        logger.error(f"[INIT] Error ensuring approved column: {e}")
        return False
    return True


def ensure_device_columns():
    """Ensure every ``device`` ORM column exists on the prod table.

    The prod ``device`` table predates several columns the ORM selects;
    a missing column makes EVERY ``db.query(Device)`` raise
    UndefinedColumn (500/503 on pairing, heartbeat, portal devices).
    ADD COLUMN IF NOT EXISTS is a no-op once healthy.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                ALTER TABLE device
                    ADD COLUMN IF NOT EXISTS device_uuid VARCHAR(255),
                    ADD COLUMN IF NOT EXISTS device_name VARCHAR(255),
                    ADD COLUMN IF NOT EXISTS device_type VARCHAR(50) DEFAULT 'thoth',
                    ADD COLUMN IF NOT EXISTS last_seen TIMESTAMP,
                    ADD COLUMN IF NOT EXISTS online BOOLEAN DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS ip_address VARCHAR(45),
                    ADD COLUMN IF NOT EXISTS mac_address VARCHAR(32),
                    ADD COLUMN IF NOT EXISTS battery_level INTEGER,
                    ADD COLUMN IF NOT EXISTS hardware_info TEXT
            """))
        return True
    except Exception as e:
        logger.error(f"[INIT] Error ensuring device columns: {e}")
        return False


def ensure_product_core_schema():
    """Apply small, idempotent schema improvements required by the product UI."""
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                ALTER TABLE user_account
                    ADD COLUMN IF NOT EXISTS email VARCHAR(320),
                    ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE,
                    ADD COLUMN IF NOT EXISTS supabase_auth_user_id VARCHAR(36)
            """))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_user_account_email ON user_account (lower(email)) WHERE email IS NOT NULL"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_user_account_supabase_auth ON user_account (supabase_auth_user_id) WHERE supabase_auth_user_id IS NOT NULL"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_payment_invoice ON payment (stripe_invoice_id) WHERE stripe_invoice_id IS NOT NULL"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_user_activity ON device (user_id, approved, last_seen DESC)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_file_visible_minutes ON device_file (device_id, modified_at DESC) WHERE on_device = TRUE"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_file_cloud_history ON device_file (user_id, modified_at DESC) WHERE on_cloud = TRUE"))
            conn.execute(text("ALTER TABLE device_file ADD COLUMN IF NOT EXISTS metadata_json TEXT"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS device_capture_chunk (
                    id SERIAL PRIMARY KEY,
                    device_id INTEGER NOT NULL REFERENCES device(device_id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id) ON DELETE CASCADE,
                    minute VARCHAR(13) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'loading',
                    occupied BOOLEAN,
                    frame_count INTEGER NOT NULL DEFAULT 10,
                    payload TEXT NOT NULL DEFAULT '{}',
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT uq_device_capture_chunk UNIQUE (device_id, minute, chunk_index)
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_live_chunk_device_minute ON device_capture_chunk (device_id, minute, chunk_index)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_live_chunk_user_updated ON device_capture_chunk (user_id, updated_at DESC)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS device_command (
                    id SERIAL PRIMARY KEY,
                    device_id INTEGER NOT NULL REFERENCES device(device_id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id) ON DELETE CASCADE,
                    command VARCHAR(40) NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    delivered_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    result TEXT
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_command_pending ON device_command (device_id, status, created_at)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS device_capture (
                    id SERIAL PRIMARY KEY,
                    capture_id VARCHAR(64) UNIQUE NOT NULL,
                    device_id INTEGER NOT NULL REFERENCES device(device_id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id) ON DELETE CASCADE,
                    state VARCHAR(20) NOT NULL DEFAULT 'requested',
                    sensors TEXT NOT NULL DEFAULT '[]',
                    sample_counts TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    stopped_at TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_capture_device ON device_capture (device_id, state)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_capture_user ON device_capture (user_id, created_at DESC)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS automation_key (
                    id SERIAL PRIMARY KEY,
                    key_hash VARCHAR(64) UNIQUE NOT NULL,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id) ON DELETE CASCADE,
                    name VARCHAR(120) NOT NULL DEFAULT '',
                    scopes TEXT NOT NULL DEFAULT '[]',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TIMESTAMP,
                    revoked BOOLEAN NOT NULL DEFAULT FALSE
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_automation_key_user ON automation_key (user_id, revoked)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS device_pairing (
                    id SERIAL PRIMARY KEY,
                    device_uuid VARCHAR(255) NOT NULL,
                    device_name VARCHAR(255) NOT NULL,
                    device_type VARCHAR(50) NOT NULL DEFAULT 'thoth',
                    hardware_info TEXT,
                    code_hash VARCHAR(64) NOT NULL UNIQUE,
                    secret_hash VARCHAR(64) NOT NULL UNIQUE,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    user_id INTEGER REFERENCES user_account(user_id) ON DELETE CASCADE,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    claimed_at TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_pairing_device ON device_pairing (device_uuid, status)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_device_pairing_expiry ON device_pairing (expires_at)"))
        return True
    except Exception as e:
        logger.error(f"[INIT] Error ensuring product core schema: {e}")
        return False


def ensure_context_schema():
    """Apply the context model + automation schema (Architecture §30–§35).

    Idempotent — safe to run on every startup. Mirrors run_migrations.py,
    which remains the manual fallback for one-off application.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS context_entity (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    entity_key VARCHAR(255) NOT NULL,
                    kind VARCHAR(80) NOT NULL,
                    name VARCHAR(255),
                    attributes TEXT,
                    retired_at DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, entity_key)
                );
                ALTER TABLE context_entity ADD COLUMN IF NOT EXISTS retired_at DOUBLE PRECISION;
                CREATE TABLE IF NOT EXISTS context_relationship (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    subject VARCHAR(255) NOT NULL,
                    predicate VARCHAR(80) NOT NULL,
                    object VARCHAR(255) NOT NULL,
                    valid_from DOUBLE PRECISION NOT NULL,
                    valid_until DOUBLE PRECISION,
                    confidence DOUBLE PRECISION DEFAULT 1.0,
                    source VARCHAR(255),
                    provenance TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS context_evidence (
                    id SERIAL PRIMARY KEY,
                    external_id VARCHAR(255),
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    evidence_key VARCHAR(255) NOT NULL,
                    value TEXT,
                    timestamp DOUBLE PRECISION NOT NULL,
                    source_id VARCHAR(255),
                    device_id VARCHAR(255),
                    prediction_id VARCHAR(255),
                    observation_id VARCHAR(255),
                    model_id VARCHAR(255),
                    model_version VARCHAR(80),
                    confidence DOUBLE PRECISION,
                    execution_class VARCHAR(40),
                    provenance TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                ALTER TABLE context_evidence ADD COLUMN IF NOT EXISTS external_id VARCHAR(255);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_context_evidence_external
                    ON context_evidence(user_id, external_id) WHERE external_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS context_state (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    state_key VARCHAR(255) NOT NULL,
                    entity_id VARCHAR(255) NOT NULL DEFAULT '',
                    value TEXT,
                    confidence DOUBLE PRECISION DEFAULT 1.0,
                    since DOUBLE PRECISION NOT NULL,
                    valid_until DOUBLE PRECISION,
                    evidence_ids TEXT,
                    estimator VARCHAR(255),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, state_key, entity_id)
                );
                UPDATE context_state SET entity_id = '' WHERE entity_id IS NULL;
                ALTER TABLE context_state ALTER COLUMN entity_id SET DEFAULT '';
                ALTER TABLE context_state ALTER COLUMN entity_id SET NOT NULL;
                CREATE TABLE IF NOT EXISTS context_event (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    event_key VARCHAR(255) NOT NULL,
                    event_type VARCHAR(20) NOT NULL,
                    entity_id VARCHAR(255),
                    state_id VARCHAR(255),
                    value TEXT,
                    previous_value TEXT,
                    confidence DOUBLE PRECISION,
                    timestamp DOUBLE PRECISION NOT NULL,
                    provenance TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_context_entity_user ON context_entity(user_id);
                CREATE INDEX IF NOT EXISTS idx_context_entity_key ON context_entity(entity_key);
                CREATE INDEX IF NOT EXISTS idx_context_rel_subject ON context_relationship(subject);
                CREATE INDEX IF NOT EXISTS idx_context_rel_predicate ON context_relationship(predicate);
                CREATE INDEX IF NOT EXISTS idx_context_evidence_key ON context_evidence(evidence_key);
                CREATE INDEX IF NOT EXISTS idx_context_evidence_ts ON context_evidence(timestamp);
                CREATE INDEX IF NOT EXISTS idx_context_state_key ON context_state(state_key);
                CREATE INDEX IF NOT EXISTS idx_context_event_key ON context_event(event_key);
                CREATE INDEX IF NOT EXISTS idx_context_event_ts ON context_event(timestamp);
                CREATE TABLE IF NOT EXISTS automation_rule (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    name VARCHAR(255) NOT NULL,
                    "when" TEXT NOT NULL,
                    "then" TEXT NOT NULL,
                    cooldown_s DOUBLE PRECISION DEFAULT 0,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_automation_rule_user ON automation_rule(user_id);
            """))
        return True
    except Exception as e:
        logger.error(f"[INIT] Error ensuring context schema: {e}")
        return False


def ensure_face_schema():
    """Apply the face-asset schema (face_basis + person_asset).

    Idempotent — mirrors run_migrations.py. Without this the /v1/faces
    endpoints 500 on a fresh DB.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS face_basis (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    name VARCHAR(255) NOT NULL DEFAULT 'default',
                    image_size INTEGER NOT NULL DEFAULT 64,
                    n_components INTEGER NOT NULL DEFAULT 0,
                    max_distance DOUBLE PRECISION DEFAULT 0,
                    data BYTEA NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, name)
                );
                CREATE INDEX IF NOT EXISTS idx_face_basis_user ON face_basis(user_id);
                CREATE TABLE IF NOT EXISTS person_asset (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    name VARCHAR(255) NOT NULL,
                    basis_id INTEGER NOT NULL REFERENCES face_basis(id),
                    projection TEXT NOT NULL,
                    photo BYTEA,
                    photo_mime VARCHAR(64),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_person_asset_user ON person_asset(user_id);
            """))
        return True
    except Exception as e:
        logger.error(f"[INIT] Error ensuring face schema: {e}")
        return False


def ensure_device_deployment_table():
    """Ensure the device_deployment table exists for pull-based model delivery."""
    try:
        direct_engine = create_engine(
            DATABASE_URL,
            pool_size=1,
            max_overflow=0,
            pool_timeout=10,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "sslmode": "require",
                "options": "-c statement_timeout=10000"
            }
        )
        with direct_engine.connect() as conn:
            table_check = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'device_deployment'
                )
            """)).scalar()
            if not table_check:
                logger.info("[INIT] Creating device_deployment table")
                conn.execute(text("""
                    CREATE TABLE device_deployment (
                        id SERIAL PRIMARY KEY,
                        deployment_id VARCHAR(255) UNIQUE NOT NULL,
                        device_uuid VARCHAR(255) NOT NULL,
                        model_id INTEGER NOT NULL,
                        user_id INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        status VARCHAR(50) NOT NULL DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        delivered_at TIMESTAMP
                    )
                """))
                conn.execute(text("CREATE INDEX idx_device_deployment_device ON device_deployment(device_uuid)"))
                conn.execute(text("CREATE INDEX idx_device_deployment_status ON device_deployment(status)"))
                conn.commit()
                logger.info("[INIT] device_deployment table created")
            else:
                logger.info("[INIT] device_deployment table already exists")
    except Exception as e:
        logger.error(f"[INIT] Error creating device_deployment table: {e}")
        return False
    return True


def ensure_node_channel_schema():
    """Apply the node↔Brain channel schema (plans/CONTRACT.md §2–§4).

    Idempotent — safe to run on every startup. Mirrors run_migrations.py.
    Without this the /v1/node/ws, /v1/events, and /v1/usage endpoints 500
    on a fresh DB.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS node_event (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    device_id VARCHAR(255) NOT NULL,
                    kind VARCHAR(80) NOT NULL,
                    data TEXT,
                    ts DOUBLE PRECISION NOT NULL,
                    external_id VARCHAR(255),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, device_id, external_id)
                );
                CREATE INDEX IF NOT EXISTS idx_node_event_user ON node_event(user_id);
                CREATE INDEX IF NOT EXISTS idx_node_event_device ON node_event(device_id);
                CREATE INDEX IF NOT EXISTS idx_node_event_kind ON node_event(kind);
                CREATE INDEX IF NOT EXISTS idx_node_event_ts ON node_event(ts);
                CREATE INDEX IF NOT EXISTS idx_node_event_created ON node_event(created_at);
                CREATE TABLE IF NOT EXISTS node_room (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    device_id VARCHAR(255) NOT NULL UNIQUE,
                    doc TEXT NOT NULL DEFAULT '{}',
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_node_room_user ON node_room(user_id);
                CREATE TABLE IF NOT EXISTS api_usage (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
                    device_id VARCHAR(255) NOT NULL,
                    ts DOUBLE PRECISION NOT NULL,
                    source VARCHAR(40) NOT NULL DEFAULT 'api',
                    kind VARCHAR(40) NOT NULL,
                    model_id VARCHAR(255),
                    latency_ms DOUBLE PRECISION,
                    tokens INTEGER,
                    meta TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_api_usage_user ON api_usage(user_id);
                CREATE INDEX IF NOT EXISTS idx_api_usage_device ON api_usage(device_id);
                CREATE INDEX IF NOT EXISTS idx_api_usage_ts ON api_usage(ts);
                CREATE INDEX IF NOT EXISTS idx_api_usage_kind ON api_usage(kind);
                CREATE INDEX IF NOT EXISTS idx_api_usage_source ON api_usage(source);
            """))
        return True
    except Exception as e:
        logger.error(f"[INIT] Error ensuring node channel schema: {e}")
        return False


def initialize_database():
    """Initialize all required database tables."""
    logger.info("[INIT] Starting database initialization")
    
    try:
        results = {
            "product_core": ensure_product_core_schema(),
            "trained_model": ensure_trained_model_table(),
            "device.approved": ensure_approved_column(),
            "device.columns": ensure_device_columns(),
            "device_deployment": ensure_device_deployment_table(),
            "context_schema": ensure_context_schema(),
            "face_schema": ensure_face_schema(),
            "node_channel": ensure_node_channel_schema(),
        }
        failed = [name for name, succeeded in results.items() if not succeeded]
        if failed:
            logger.warning("[INIT] Database initialization incomplete; failed checks: %s", ", ".join(failed))
            return False
        logger.info("[INIT] Database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"[INIT] Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    initialize_database()
