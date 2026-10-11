"""One-off: full actuator inventory per approved device on prod."""
import json
import psycopg2

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
cur = conn.cursor()
cur.execute("""
    SELECT device_name, online, hardware_info
    FROM device WHERE approved=true ORDER BY device_name
""")
for name, online, hw in cur.fetchall():
    try:
        info = json.loads(hw) if hw else {}
    except Exception:
        info = {}
    acts = [(a.get("id"), a.get("kind"), a.get("name"))
            for a in info.get("actuators", []) if isinstance(a, dict)]
    print(name, "online:", online)
    for a in acts:
        print("   ", a)
