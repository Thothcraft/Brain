"""One-off: create the missing automation tables on prod (idempotent)."""
import psycopg2

DDL = """
CREATE TABLE IF NOT EXISTS automation_rule_state (
    id SERIAL PRIMARY KEY,
    rule_id INTEGER NOT NULL UNIQUE REFERENCES automation_rule(id),
    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
    matched BOOLEAN DEFAULT FALSE,
    last_fired_at TIMESTAMPTZ,
    last_execution_id VARCHAR(64),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_automation_rule_state_user ON automation_rule_state(user_id);
CREATE TABLE IF NOT EXISTS automation_execution (
    id SERIAL PRIMARY KEY,
    execution_id VARCHAR(64) UNIQUE NOT NULL,
    rule_id INTEGER NOT NULL REFERENCES automation_rule(id),
    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
    rule_name VARCHAR(255) NOT NULL,
    trigger_state_id INTEGER,
    trigger_event_id INTEGER,
    trigger_snapshot TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'fired',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_automation_execution_user ON automation_execution(user_id);
CREATE INDEX IF NOT EXISTS idx_automation_execution_rule ON automation_execution(rule_id);
CREATE TABLE IF NOT EXISTS action_request (
    id SERIAL PRIMARY KEY,
    action_id VARCHAR(64) UNIQUE NOT NULL,
    execution_id VARCHAR(64),
    user_id INTEGER NOT NULL REFERENCES user_account(user_id),
    device_id VARCHAR(255) NOT NULL,
    actuator_id VARCHAR(255) NOT NULL,
    operation VARCHAR(64) NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    origin VARCHAR(255) NOT NULL DEFAULT '',
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    dispatched_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    last_error TEXT,
    result_json TEXT,
    correlation TEXT
);
CREATE INDEX IF NOT EXISTS idx_action_request_user ON action_request(user_id);
CREATE INDEX IF NOT EXISTS idx_action_request_status ON action_request(status);
CREATE INDEX IF NOT EXISTS idx_action_request_execution ON action_request(execution_id);
"""

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
conn.autocommit = True
cur = conn.cursor()
cur.execute(DDL)
cur.execute("""
    SELECT tablename FROM pg_tables
    WHERE tablename LIKE 'automation%' OR tablename LIKE 'action%'
    ORDER BY tablename
""")
print([r[0] for r in cur.fetchall()])
