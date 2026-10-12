#!/bin/bash
set -e
echo '--- rewrite ha.conf ---'
cat >/tmp/ha.conf <<'EOF'
[Service]
Environment=HA_URL=http://127.0.0.1:8123
Environment=HA_TOKEN=eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJpc3MiOiAiZmYwYWYyMmYwOGZlNDZiYzk2MDI0MWYxMjg5NjI0MmYiLCAiaWF0IjogMTc5MDMxNjU4MCwgImV4cCI6IDIxMDU2NzY1ODB9.-7zlKHH6IzmWVxUfgdN66fL4GbtXc-rwHmHZrBDoLfg
Environment=HA_ENTITIES=light.gad_room_gad_room,light.gad_room_hue
EOF
echo password | sudo -S mv /tmp/ha.conf /etc/systemd/system/thoth.service.d/ha.conf
echo password | sudo -S cat /etc/systemd/system/thoth.service.d/ha.conf | sed 's/HA_TOKEN=.*/HA_TOKEN=<redacted>/'

echo '--- restart ---'
echo password | sudo -S systemctl daemon-reload
echo password | sudo -S systemctl restart thoth.service
sleep 10
PID=$(pgrep -f 'thoth daemon' | head -1)
echo "daemon pid: $PID"
echo '--- env ---'
cat /proc/$PID/environ | tr '\0' '\n' | grep '^HA_' | sed 's/HA_TOKEN=.*/HA_TOKEN=<set>/'
echo '--- actuators ---'
for i in 1 2 3; do
  OUT=$(curl -sm5 -H 'Authorization: Bearer 07OUVzEOfBlZcYk4mhW_xKlMjuz1Brup' \
    http://127.0.0.1:5000/api/v1/actuators 2>/dev/null) && break
  sleep 3
done
echo "$OUT" | python3 -c 'import json,sys; [print(a["id"], a["kind"], a.get("adapter")) for a in json.load(sys.stdin)["actuators"]]'
