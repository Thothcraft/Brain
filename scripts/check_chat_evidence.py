"""What does the chat agent's bundle actually see right now on prod?

- Active context states (occupancy/prediction — the "is it empty" answer)
- Recent evidence keys (radar/csi/ble/motion descriptors feeding bundles)
"""
import psycopg2

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
cur = conn.cursor()

cur.execute("""
    SELECT state_key, entity_id, substring(value, 1, 80),
           estimator, confidence,
           to_char(to_timestamp(since), 'MM-DD HH24:MI:SS') since,
           to_char(to_timestamp(valid_until), 'MM-DD HH24:MI:SS') exp,
           to_char(now(), 'MM-DD HH24:MI:SS') now
    FROM context_state
    ORDER BY updated_at DESC LIMIT 30
""")
print("== ACTIVE STATES ==")
for r in cur.fetchall():
    print(" ", r)

cur.execute("""
    SELECT evidence_key, device_id, count(*),
           to_char(to_timestamp(max(timestamp)), 'MM-DD HH24:MI:SS') latest,
           round(extract(epoch from now()) - max(timestamp))::int as age_s
    FROM context_evidence
    WHERE timestamp > extract(epoch from now()) - 21600
    GROUP BY evidence_key, device_id
    ORDER BY max(timestamp) DESC LIMIT 40
""")
print("\n== EVIDENCE (last 6h) ==")
for r in cur.fetchall():
    print(" ", r)
