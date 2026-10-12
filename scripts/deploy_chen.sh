#!/bin/bash
set -e
cd /home/gad/thoth
git pull --ff-only 2>&1 | tail -3
echo password | sudo -S systemctl restart thoth.service
sleep 10
echo '--- service ---'
systemctl is-active thoth.service
echo '--- actuators ---'
for i in 1 2 3 4 5; do
  OUT=$(curl -sm5 -H 'Authorization: Bearer 07OUVzEOfBlZcYk4mhW_xKlMjuz1Brup' \
    http://127.0.0.1:5000/api/v1/actuators 2>/dev/null) && [ -n "$OUT" ] && break
  sleep 3
done
echo "$OUT" | python3 -c 'import json,sys; [print(a["id"], a["kind"], a.get("adapter")) for a in json.load(sys.stdin)["actuators"]]'
