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
    """
    
    try:
        from server.db import engine
        with engine.connect() as conn:
            conn.execute(migration_sql)
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
