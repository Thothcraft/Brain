"""One-off: device registry duplicates on prod."""
import psycopg2

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
cur = conn.cursor()
cur.execute("""
    SELECT device_id, device_uuid, device_name, device_type,
           mac_address, approved, online, last_seen
    FROM device ORDER BY device_name
""")
for r in cur.fetchall():
    print(r)
