#!/bin/bash
echo password | sudo -S cat /etc/systemd/system/thoth.service.d/ha.conf
echo '--- service ---'
systemctl is-active thoth.service
PID=$(pgrep -f 'thoth daemon' | head -1)
echo "daemon pid: $PID"
echo '--- env ---'
echo password | sudo -S tr '\0' '\n' </proc/$PID/environ | grep '^HA_' | sed 's/HA_TOKEN=.*/HA_TOKEN=<set>/'
echo '--- actuators ---'
curl -sm5 -H 'Authorization: Bearer 07OUVzEOfBlZcYk4mhW_xKlMjuz1Brup' \
  http://127.0.0.1:5000/api/v1/actuators \
  | python3 -c 'import json,sys; [print(a["id"], a["kind"], a.get("adapter")) for a in json.load(sys.stdin)["actuators"]]'
echo '--- last daemon log ---'
echo password | sudo -S journalctl -u thoth.service --since '3 min ago' -n 20 --no-pager | tail -20
