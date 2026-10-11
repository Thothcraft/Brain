"""One-off: clean duplicate phone/laptop device+entity rows on prod.

Registry: keep the freshest approved phone row (1757 b62b6a75, online);
unapprove stale same-phone rows. Context entities: retire ghosts and the
bogus place:thoth-laptop; strip 'thoth-toronto' from april's aliases.
"""
import json
import psycopg2

STALE_DEVICES = [1752, 1754, 1755, 1756]
RETIRE_ENTITIES = [
    "device:6ee6922b-91f0-4343-ad3d-97653354e1da",
    "device:75d66ed3-7ca2-49e9-9386-b6f22577ee73",
    "device:95b9d4b3-ae9f-4889-bbd4-ef3b823a0cee",
    "device:f5c15bda-8013-46b1-a16b-4e66e4b86318",
    "device:c1046bd3-dc05-5a34-a502-32d5bbb859a8",
    "device:c1046bd3-dc05-5a502-32d5bbb859a8",
    "place:thoth-laptop",
]
CANON_PHONE_KEY = "device:b62b6a75-848b-4bf9-b327-855b20f43cc2"

conn = psycopg2.connect(
    host="aws-0-us-east-1.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.otjjjagmwzswbinwxmfw",
    password="EgyptCanada2021..", connect_timeout=10)
cur = conn.cursor()

cur.execute("UPDATE device SET approved=false WHERE device_id = ANY(%s)",
            (STALE_DEVICES,))
print("unapproved device rows:", cur.rowcount)

cur.execute("""
    UPDATE context_entity SET retired_at = EXTRACT(EPOCH FROM NOW())
    WHERE entity_key = ANY(%s) AND retired_at IS NULL
""", (RETIRE_ENTITIES,))
print("retired entities:", cur.rowcount)

# april aliases: drop thoth-toronto
cur.execute("""
    SELECT id, attributes FROM context_entity
    WHERE entity_key = 'device:168477ac-fbb3-53f6-9c9a-3c7d894ff545'
""")
row = cur.fetchone()
if row:
    attrs = json.loads(row[1] or "{}")
    aliases = [a for a in attrs.get("aliases", []) if a != "thoth-toronto"]
    if aliases != attrs.get("aliases"):
        attrs["aliases"] = aliases
        cur.execute("UPDATE context_entity SET attributes=%s WHERE id=%s",
                    (json.dumps(attrs), row[0]))
        print("april aliases ->", aliases)

# canonical phone entity: make sure 'phone:gadgad'/'phone:this' resolve as
# aliases so future matches land on the live device
cur.execute("""
    SELECT id, attributes FROM context_entity
    WHERE entity_key = %s
""", (CANON_PHONE_KEY,))
row = cur.fetchone()
if row:
    attrs = json.loads(row[1] or "{}")
    aliases = set(attrs.get("aliases", []))
    aliases.update({"phone:this", "phone:gadgad"})
    attrs["aliases"] = sorted(a for a in aliases if a)
    cur.execute("UPDATE context_entity SET attributes=%s WHERE id=%s",
                (json.dumps(attrs), row[0]))
    print("canonical phone aliases ->", attrs["aliases"])

conn.commit()
print("committed")
