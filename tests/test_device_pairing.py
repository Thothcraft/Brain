import asyncio
from datetime import datetime, timedelta
from unittest import mock

from server.db import Device, DevicePairing, User
from server.endpoints.device_endpoints import (
    claim_device_pairing,
    device_pairing_status,
    start_device_pairing,
)
from server.endpoints.models import DevicePairingClaimRequest, DevicePairingStartRequest


class _Query:
    def __init__(self, records):
        self.records = records

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.records[0] if self.records else None

    def update(self, values, synchronize_session=False):
        for record in self.records:
            for key, value in values.items():
                setattr(record, key, value)
        return len(self.records)


class _Database:
    def __init__(self, records=None):
        self.records = records or {}

    def query(self, model):
        return _Query(self.records.setdefault(model, []))

    def add(self, value):
        self.records.setdefault(type(value), []).append(value)

    def commit(self):
        pass


def test_device_can_start_and_account_can_claim_pairing():
    database = _Database()
    started = asyncio.run(start_device_pairing(DevicePairingStartRequest(
        device_id="physical-device",
        device_name="Bedroom Thoth",
        hardware_info={"hostname": "thoth"},
    ), database))
    pairing = database.records[DevicePairing][0]
    # The endpoint stores only hashes; use the returned code by assigning its
    # hash to the in-memory record exactly as a database lookup would.
    assert len(started["code"]) == 8
    user = User(userId=7, username="owner", email="owner@example.com", role=0)

    claimed = asyncio.run(claim_device_pairing(
        DevicePairingClaimRequest(code=started["code"]), user, database,
    ))

    assert claimed["status"] == "paired"
    assert pairing.user_id == 7
    assert database.records[Device][0].device_name == "Bedroom Thoth"


def test_only_initiating_device_receives_paired_token():
    user = User(userId=7, username="owner", email="owner@example.com", role=0)
    device = Device(userId=7, device_uuid="device-id", device_name="Bedroom Thoth", device_type="thoth")
    pairing = DevicePairing(
        device_uuid="device-id",
        device_name="Bedroom Thoth",
        device_type="thoth",
        code_hash="code",
        secret_hash="secret-hash",
        status="claimed",
        user_id=7,
        expires_at=datetime.utcnow() + timedelta(minutes=5),
    )
    database = _Database({DevicePairing: [pairing], User: [user], Device: [device]})

    with mock.patch("server.endpoints.device_endpoints._pairing_hash", return_value="secret-hash"), \
         mock.patch("server.auth.create_access_token", return_value="device-token"):
        result = asyncio.run(device_pairing_status("private-secret", database))

    assert result["status"] == "paired"
    assert result["access_token"] == "device-token"
