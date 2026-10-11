"""Live agent probe — seeds a realistic room (watch unworn on desk,
laptop screen on Chrome/models + idle input, phone app in foreground,
empty-room radar stats) and asks the assistant the questions a user
would. Requires OPENAI_API_KEY in .env. Run:  python _probe_agent.py
"""
import json
import os
import sys
import time

# Load .env before importing server modules.
for line in open(os.path.join(os.path.dirname(__file__), ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import (Base, ContextEvidence, ContextState, Device, User,
                       get_db)
import server.v1.chat as chat_mod

NOW = time.time()

engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                       poolclass=StaticPool)
Base.metadata.create_all(engine)
factory = sessionmaker(bind=engine, expire_on_commit=False)
session = factory()

user = User(userId=1, username="gadgad", email="g@e.invalid",
            hashed_password="x", role=0, plan="free")
session.add(user)
session.add(Device(userId=1, device_uuid="lap-1", device_name="gad-laptop",
                   device_type="thoth", mac_address="aa:bb:cc:00:11:22"))
session.add(Device(userId=1, device_uuid="phone-1", device_name="gad-phone",
                   device_type="android"))
session.add(Device(userId=1, device_uuid="watch-1", device_name="PineTime",
                   device_type="pinetime", mac_address="d8:3a:dd:99:88:77"))
session.commit()


def state(key, entity, value, conf=0.9, estimator="node"):
    session.add(ContextState(
        user_id=1, state_key=key, entity_id=entity,
        value=json.dumps(value), confidence=conf, estimator=estimator,
        since=NOW - 30))


# --- the room right now -----------------------------------------------------
state("watch.link.v1", "device:watch-1",
      {"connected": True, "rssi": -49}, estimator="watch.relay")
state("watch.worn.v1", "device:watch-1",
      {"worn": False, "acc_std": 0.0041, "n": 150}, estimator="watch.relay")
state("proximity.v1", "device:watch-1",
      {"distance_m": 1.9, "rssi": -49}, estimator="watch.relay")
state("screen.v1", "device:lap-1", {"on": True}, estimator="pc.agent")
state("active_window.v1", "device:lap-1",
      {"app": "chrome", "title": "Models - hub.thothcraft.com",
       "url": "https://hub.thothcraft.com/models"},
      estimator="pc.agent")
state("input.idle.v1", "device:lap-1", {"idle_s": 840},
      estimator="pc.agent")
state("app.focus.v1", "device:phone-1",
      {"app": "thothcraft", "foreground": True},
      estimator="mobile.lifecycle")
state("occupancy.v1", "place:home", {"occupied": False}, conf=0.72)

# Empty-room radar/CSI evidence (matches the reference empty profile:
# snr mean ~7.6 tight, no >15 dB tail spikes, low csi variance).
for i in range(30):
    ts = NOW - 30 + i
    session.add(ContextEvidence(
        user_id=1, evidence_key="radar.snr", device_id="lap-1",
        timestamp=ts, confidence=0.9,
        value=json.dumps({"snr_db": 7.4 + (i % 5) * 0.12,
                          "snr_max": 8.6 + (i % 3) * 0.4})))
    session.add(ContextEvidence(
        user_id=1, evidence_key="csi.variance", device_id="lap-1",
        timestamp=ts, confidence=0.9,
        value=json.dumps({"var": 0.031 + (i % 4) * 0.002})))
session.commit()

app = FastAPI()
app.include_router(chat_mod.router, prefix="/v1")
app.dependency_overrides[get_current_user] = lambda: user
app.dependency_overrides[get_db] = lambda: session
client = TestClient(app)

QUESTIONS = [
    "Is my watch on my wrist right now?",
    "Is my laptop screen on? What is being done on my laptop?",
    "Is anyone in the room right now?",
    "What is happening on my phone?",
    "Where are my devices and who do they belong to?",
]

tier = sys.argv[1] if len(sys.argv) > 1 else "4"
print(f"== model pick: {tier} ==\n")
for q in QUESTIONS:
    r = client.post("/v1/chat", json={"message": q, "model": tier,
                                      "history": []})
    print(f"Q: {q}")
    if r.status_code != 200:
        print(f"   HTTP {r.status_code}: {r.text[:300]}\n")
        continue
    body = r.json()
    print(f"A: {body.get('answer')}")
    for a in body.get("assessments") or []:
        print(f"   p={a.get('probability')}  {a.get('claim')}"
              f"  [{a.get('basis')}]")
    print()
