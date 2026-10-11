"""One-off: hardware_info actuator inventory per device on prod."""
import json
import psycopg2

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
cur = conn.cursor()
cur.execute("""
    SELECT device_name, device_uuid, online, hardware_info
    FROM device WHERE approved = true ORDER BY device_name
""")
for name, uuid, online, hw in cur.fetchall():
    try:
        info = json.loads(hw) if hw else {}
    except Exception:
        info = {}
    print(f"{name} ({uuid[:8]}) online={online}")
    print("  local_api:", info.get("local_api"))
    print("  actuators:", json.dumps(info.get("actuators"))[:300])
    print("  keys:", list(info.keys()))
