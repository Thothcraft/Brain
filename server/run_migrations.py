#!/usr/bin/env python3
"""Run Supabase migrations and seed data."""

import os
import sys
from pathlib import Path

# Add server root to path
server_root = Path(__file__).parent
sys.path.insert(0, str(server_root))

from server.db import get_db, User
from server.auth import get_password_hash

def run_migrations():
    """Execute SQL migrations on Supabase."""
    print("[migrations] Running database migrations...")
    
    # Read migration SQL
    migration_sql = """
    -- Add new columns to user_account table
    ALTER TABLE user_account 
    ADD COLUMN IF NOT EXISTS plan VARCHAR(50) DEFAULT 'free',
    ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS stripe_subscription_id VARCHAR(255),
    ADD COLUMN IF NOT EXISTS plan_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS org_name VARCHAR(255);

    -- Create new tables
    CREATE TABLE IF NOT EXISTS org_membership (
        id SERIAL PRIMARY KEY,
        org_id INTEGER REFERENCES user_account(user_id) ON DELETE CASCADE,
        member_id INTEGER REFERENCES user_account(user_id) ON DELETE CASCADE,
        status VARCHAR(20) DEFAULT 'pending',
        invited_at TIMESTAMPTZ DEFAULT NOW(),
        approved_at TIMESTAMPTZ,
        invite_code VARCHAR(50),
        UNIQUE(org_id, member_id)
    );

    CREATE TABLE IF NOT EXISTS invite_code (
        id SERIAL PRIMARY KEY,
        code VARCHAR(50) UNIQUE NOT NULL,
        org_id INTEGER REFERENCES user_account(user_id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        expires_at TIMESTAMPTZ,
        max_uses INTEGER DEFAULT 100,
        uses_count INTEGER DEFAULT 0,
        is_active BOOLEAN DEFAULT TRUE
    );

    CREATE TABLE IF NOT EXISTS lab (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        sensor_type VARCHAR(50) NOT NULL,
        difficulty VARCHAR(20) DEFAULT 'beginner',
        questions TEXT NOT NULL,
        max_score INTEGER DEFAULT 100,
        created_by INTEGER REFERENCES user_account(user_id),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        is_published BOOLEAN DEFAULT TRUE
    );

    CREATE TABLE IF NOT EXISTS lab_submission (
        id SERIAL PRIMARY KEY,
        lab_id INTEGER REFERENCES lab(id) ON DELETE CASCADE,
        user_id INTEGER REFERENCES user_account(user_id) ON DELETE CASCADE,
        org_id INTEGER REFERENCES user_account(user_id),
        answers TEXT NOT NULL,
        score NUMERIC,
        max_score INTEGER,
        submitted_at TIMESTAMPTZ DEFAULT NOW(),
        graded_at TIMESTAMPTZ,
        feedback TEXT,
        UNIQUE(lab_id, user_id)
    );

    CREATE TABLE IF NOT EXISTS payment (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES user_account(user_id) ON DELETE CASCADE,
        stripe_payment_intent VARCHAR(255),
        stripe_invoice_id VARCHAR(255),
        amount INTEGER NOT NULL,
        currency VARCHAR(10) DEFAULT 'usd',
        plan VARCHAR(50),
        status VARCHAR(50) DEFAULT 'pending',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_org_membership_org_id ON org_membership(org_id);
    CREATE INDEX IF NOT EXISTS idx_org_membership_member_id ON org_membership(member_id);
    CREATE INDEX IF NOT EXISTS idx_org_membership_status ON org_membership(status);
    CREATE INDEX IF NOT EXISTS idx_invite_code_code ON invite_code(code);
    CREATE INDEX IF NOT EXISTS idx_invite_code_org_id ON invite_code(org_id);
    CREATE INDEX IF NOT EXISTS idx_lab_sensor_type ON lab(sensor_type);
    CREATE INDEX IF NOT EXISTS idx_lab_submission_lab_id ON lab_submission(lab_id);
    CREATE INDEX IF NOT EXISTS idx_lab_submission_user_id ON lab_submission(user_id);
    CREATE INDEX IF NOT EXISTS idx_payment_user_id ON payment(user_id);
    CREATE INDEX IF NOT EXISTS idx_payment_status ON payment(status);

    -- Spatial layer: named areas, zones, device placement
    CREATE TABLE IF NOT EXISTS space (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES user_account(user_id),
        parent_id INTEGER REFERENCES space(id),
        name VARCHAR(255) NOT NULL,
        floor_plan_file_id INTEGER REFERENCES file(file_id),
        width_m DOUBLE PRECISION,
        height_m DOUBLE PRECISION,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS zone (
        id SERIAL PRIMARY KEY,
        space_id INTEGER NOT NULL REFERENCES space(id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        polygon_json TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS device_placement (
        id SERIAL PRIMARY KEY,
        device_id INTEGER NOT NULL UNIQUE REFERENCES device(device_id) ON DELETE CASCADE,
        space_id INTEGER NOT NULL REFERENCES space(id) ON DELETE CASCADE,
        x DOUBLE PRECISION NOT NULL DEFAULT 0,
        y DOUBLE PRECISION NOT NULL DEFAULT 0,
        rotation_deg DOUBLE PRECISION NOT NULL DEFAULT 0,
        fov_deg DOUBLE PRECISION NOT NULL DEFAULT 90,
        range_m DOUBLE PRECISION NOT NULL DEFAULT 8,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_space_user_id ON space(user_id);
    CREATE INDEX IF NOT EXISTS idx_zone_space_id ON zone(space_id);
    CREATE INDEX IF NOT EXISTS idx_device_placement_space ON device_placement(space_id);

    -- Processor-ecosystem metadata on trained_model
    ALTER TABLE trained_model
    ADD COLUMN IF NOT EXISTS processor_type VARCHAR(20) DEFAULT 'torchscript',
    ADD COLUMN IF NOT EXISTS sensor VARCHAR(50),
    ADD COLUMN IF NOT EXISTS task VARCHAR(50),
    ADD COLUMN IF NOT EXISTS visibility VARCHAR(20) DEFAULT 'private',
    ADD COLUMN IF NOT EXISTS registry_name VARCHAR(255);
    CREATE INDEX IF NOT EXISTS idx_trained_model_registry_name ON trained_model(registry_name);

    -- Context model (Architecture §30–§35)
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
    ALTER TABLE context_evidence ADD COLUMN IF NOT EXISTS external_id VARCHAR(255);
    CREATE UNIQUE INDEX IF NOT EXISTS uq_context_evidence_external
        ON context_evidence(user_id, external_id) WHERE external_id IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_context_evidence_key ON context_evidence(evidence_key);
    CREATE INDEX IF NOT EXISTS idx_context_evidence_ts ON context_evidence(timestamp);
    CREATE INDEX IF NOT EXISTS idx_context_state_key ON context_state(state_key);
    UPDATE context_state SET entity_id = '' WHERE entity_id IS NULL;
    ALTER TABLE context_state ALTER COLUMN entity_id SET DEFAULT '';
    ALTER TABLE context_state ALTER COLUMN entity_id SET NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_context_event_key ON context_event(event_key);
    CREATE INDEX IF NOT EXISTS idx_context_event_ts ON context_event(timestamp);

    -- Automation rules (evaluated server-side by Brain's AutomationEngine)
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

    -- Node↔Brain channel (plans/CONTRACT.md §2–§4)
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
    """

    try:
        from sqlalchemy import text
        from server.db import engine
        with engine.connect() as conn:
            conn.execute(text(migration_sql))
            conn.commit()
        print("[migrations] ✅ Migrations completed successfully")
    except Exception as e:
        print(f"[migrations] ❌ Migration failed: {e}")
        raise

def seed_admin():
    """Create default admin user."""
    print("[seed] Creating admin user...")
    db = next(get_db())
    try:
        existing = db.query(User).filter(User.username == "admin").first()
        if existing:
            print("[seed] Admin user already exists")
            return

        admin_user = User(
            username="admin",
            hashed_password=get_password_hash("password"),
            role=1,  # admin
            plan="organization",
            org_name="Thothcraft Admin",
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        print(f'[seed] ✅ Created admin user: userId={admin_user.userId}, username=admin, password=password')
    finally:
        db.close()

if __name__ == "__main__":
    run_migrations()
    seed_admin()
    print("\n✅ Setup complete! You can now:")
    print("   1. Log in as admin/password")
    print("   2. Configure Stripe keys")
    print("   3. Create labs via admin dashboard")
