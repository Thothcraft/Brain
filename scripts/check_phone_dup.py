"""One-off: inspect phone:* entities on prod."""
import psycopg2

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
cur = conn.cursor()
cur.execute("""
    SELECT entity_key, name, kind, retired_at IS NULL, attributes
    FROM context_entity
    WHERE entity_key LIKE 'phone%' OR entity_key LIKE 'device%phone%'
       OR name ILIKE '%phone%' OR name ILIKE '%thoth%'
    ORDER BY entity_key
""")
for r in cur.fetchall():
    print(r[0], "|", r[1], "|", r[2], "| live:", r[3], "|", (r[4] or "")[:200])
