"""Dump the exact prompt the chat model sees for the occupancy
question, plus the raw model response — to two files:
    _probe_occupancy_prompt.txt    (system + user messages verbatim)
    _probe_occupancy_response.txt  (raw completion JSON)
Run:  python _probe_occupancy_dump.py
"""
import json
import os
import time

for line in open(os.path.join(os.path.dirname(__file__), ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.resources.chat.completions import Completions
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.auth import get_current_user
from server.db import (Base, ContextEvidence, ContextState, Device, User,
                       get_db)
import server.v1.chat as chat_mod

# --- capture the real OpenAI call --------------------------------------------
captured = {}
_orig_create = Completions.create


def _spy(self, *a, **kw):
    resp = _orig_create(self, *a, **kw)
    captured["messages"] = kw.get("messages", [])
    captured["kwargs"] = {k: v for k, v in kw.items() if k != "messages"}
    captured["raw"] = resp.choices[0].message.content
    return resp


Completions.create = _spy

NOW = time.time()
engine = create_engine("sqlite://",
                       connect_args={"check_same_thread": False},
                       poolclass=StaticPool)
Base.metadata.create_all(engine)
session = sessionmaker(bind=engine, expire_on_commit=False)()

user = User(userId=1, username="gadgad", email="g@e.invalid",
            hashed_password="x", role=0, plan="free")
session.add(user)
session.add(Device(userId=1, device_uuid="lap-1", device_name="gad-laptop",
                   device_type="thoth"))
session.add(Device(userId=1, device_uuid="watch-1", device_name="PineTime",
                   device_type="pinetime"))
session.commit()

session.add(ContextState(
    user_id=1, state_key="occupancy.v1", entity_id="place:home",
    value=json.dumps({"occupied": False}), confidence=0.72,
    estimator="occ-v1", since=NOW - 30))
session.add(ContextState(
    user_id=1, state_key="watch.worn.v1", entity_id="device:watch-1",
    value=json.dumps({"worn": False, "acc_std": 0.0041, "n": 150}),
    confidence=0.9, estimator="watch.relay", since=NOW - 30))
# empty-room radar + csi evidence
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

r = client.post("/v1/chat", json={
    "message": "Is anyone in the room right now?",
    "model": "4", "history": []})
print("HTTP", r.status_code)

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "_probe_occupancy_prompt.txt"), "w",
          encoding="utf-8") as f:
    f.write("=== request kwargs (model, response_format, etc.) ===\n")
    f.write(json.dumps(captured["kwargs"], indent=2, default=str))
    f.write("\n\n=== messages sent to the model ===\n")
    for m in captured["messages"]:
        f.write(f"\n----- {m['role']} -----\n{m['content']}\n")
with open(os.path.join(here, "_probe_occupancy_response.txt"), "w",
          encoding="utf-8") as f:
    f.write("=== raw completion content ===\n")
    f.write(captured["raw"])
    f.write("\n\n=== parsed API response to the app ===\n")
    f.write(json.dumps(r.json(), indent=2))
print("wrote _probe_occupancy_prompt.txt + _probe_occupancy_response.txt")
