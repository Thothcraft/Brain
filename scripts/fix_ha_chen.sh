#!/bin/bash
# Fix HA actuator env on thoth-chen — drop-in was attached to
# thoth-node.service but the daemon runs under thoth.service.
set -e
TOKEN='eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.eyJpc3MiOiAiZmYwYWYyMmYwOGZlNDZiYzk2MDI0MWYxMjg5NjI0MmYiLCAiaWF0IjogMTc5MDMxNjU4MCwgImV4cCI6IDIxMDU2NzY1ODB9.-7zlKHH6IzmWVxUfgdN66fL4GbtXc-rwHmHZrBDoLfg'

echo '--- lights on HA ---'
curl -sm5 -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8123/api/states \
  | python3 -c 'import json,sys; [print(s["entity_id"], s["state"]) for s in json.load(sys.stdin) if s["entity_id"].startswith("light.")]' || true

echo '--- write drop-in for thoth.service ---'
echo password | sudo -S mkdir -p /etc/systemd/system/thoth.service.d
cat /etc/systemd/system/thoth-node.service.d/ha.conf \
  | sed 's/^Environment=HA_ENTITIES=.*/Environment=HA_ENTITIES=light.gad_room_gad_room/' \
  | ( echo password | sudo -S tee /etc/systemd/system/thoth.service.d/ha.conf >/dev/null )
sudo -n cat /etc/systemd/system/thoth.service.d/ha.conf || echo password | sudo -S cat /etc/systemd/system/thoth.service.d/ha.conf

echo '--- reload + restart thoth.service ---'
echo password | sudo -S systemctl daemon-reload
echo password | sudo -S systemctl restart thoth.service
sleep 8
echo '--- actuators after restart ---'
TOKEN_NODE='07OUVzEOfBlZcYk4mhW_xKlMjuz1Brup'
curl -sm5 -H "Authorization: Bearer $TOKEN_NODE" \
  http://127.0.0.1:5000/api/v1/actuators \
  | python3 -m json.tool | head -60
