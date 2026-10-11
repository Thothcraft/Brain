"""One-off: upgrade watch-near-green to a multi-action rule on prod."""
import json
import psycopg2

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
cur = conn.cursor()
then = [
    # resolves to the HA light when chen advertises it again; queued
    # until then — retry endpoint redispatches later.
    {"actuator_id": "light", "operation": "set_color",
     "params": {"rgb": [0, 255, 0]}},
    # speaks on the best online node NOW (april/laptop/chen speakers)
    {"actuator_id": "speaker", "operation": "speak",
     "params": {"text": "watch nearby"}},
    {"actuator_id": "notify", "operation": "notify",
     "params": {"title": "Thoth - watch near",
                "body": "your watch is within 10 m"}},
]
cur.execute(
    "UPDATE automation_rule SET \"then\"=%s WHERE name='watch-near-green'",
    (json.dumps(then),))
print("updated:", cur.rowcount)
conn.commit()
cur.execute("SELECT \"then\" FROM automation_rule WHERE name='watch-near-green'")
print(cur.fetchone()[0])
