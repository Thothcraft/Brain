"""One-off: list automation/action/spatial tables on prod."""
import psycopg2

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
cur = conn.cursor()
cur.execute("""
    SELECT tablename FROM pg_tables
    WHERE tablename LIKE 'automation%'
       OR tablename LIKE 'action%'
       OR tablename IN ('space','zone','device_placement','person')
    ORDER BY tablename
""")
for r in cur.fetchall():
    print(r[0])
