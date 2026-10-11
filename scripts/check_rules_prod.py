"""One-off: inspect rules, states, executions, actions on prod."""
import psycopg2

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
cur = conn.cursor()

for tbl, col in [("automation_rule", "name"), ("automation_execution", "rule_name"),
                 ("action_request", "actuator_id")]:
    try:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        print(tbl, "rows:", cur.fetchone()[0])
    except Exception as e:
        print(tbl, "ERR:", e)
        conn.rollback()

cur.execute("SELECT name, enabled, \"when\", \"then\", cooldown_s FROM automation_rule")
for r in cur.fetchall():
    print("RULE:", r)

cur.execute("""
    SELECT state_key, entity_id, substring(value, 1, 120),
           to_char(to_timestamp(since), 'MM-DD HH24:MI') ts,
           valid_until
    FROM context_state ORDER BY updated_at DESC LIMIT 25
""")
for r in cur.fetchall():
    print("STATE:", r)

cur.execute("""
    SELECT event_key, substring(value, 1, 100),
           to_char(to_timestamp(timestamp), 'MM-DD HH24:MI') ts
    FROM context_event ORDER BY timestamp DESC LIMIT 15
""")
for r in cur.fetchall():
    print("EVENT:", r)

cur.execute("""
    SELECT actuator_id, operation, status, last_error,
           to_char(created_at, 'MM-DD HH24:MI') ts
    FROM action_request ORDER BY created_at DESC LIMIT 15
""")
for r in cur.fetchall():
    print("ACTION:", r)
