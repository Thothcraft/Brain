import asyncio
import uuid
from datetime import datetime, timedelta
from unittest import mock

from fastapi import HTTPException

from server.db import (
    Device, DeviceCaptureChunk, DeviceCommand, DeviceDeployment, DeviceFile,
    DevicePairing, FileDeviceUpdate, User,
)
from server.endpoints.device_endpoints import (
    claim_device_pairing,
    device_pairing_status,
    start_device_pairing,
    update_device_identity,
)
from server.endpoints.models import DevicePairingClaimRequest, DevicePairingStartRequest

# start_device_pairing normalizes request.device_id via uuid5 before
# querying — fixtures must store the normalized form to be found.
NORMALIZED_DEVICE_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "device-id"))


def _expr_value(side):
    """Extract a Python value from a SQLAlchemy expression side."""
    return getattr(side, "value", side)


def _attr_for(record, name):
    """Map a column name (``user_id``) to the ORM attribute (``userId``)."""
    if hasattr(record, name):
        return name
    try:
        from sqlalchemy import inspect as sa_inspect
        for attr in sa_inspect(type(record)).mapper.attrs:
            cols = getattr(attr, "columns", None)
            if cols and cols[0].name == name:
                return attr.key
    except Exception:
        pass
    return name


def _matches(record, expr) -> bool:
    """Evaluate a simple SQLAlchemy binary expression against a record."""
    import operator as _op
    left = getattr(expr, "left", None)
    right = getattr(expr, "right", None)
    op = getattr(expr, "operator", None)
    if left is None or op is None:
        return True
    name = getattr(left, "key", None) or getattr(left, "name", None)
    actual = getattr(record, _attr_for(record, name), None)
    expected = _expr_value(right)
    try:
        if op == _op.eq:
            return actual == expected
        if op == _op.ne:
            return actual != expected
        if op == _op.gt:
            return actual is not None and expected is not None and actual > expected
        if op == _op.ge:
            return actual is not None and expected is not None and actual >= expected
        if op == _op.lt:
            return actual is not None and expected is not None and actual < expected
        if op == _op.le:
            return actual is not None and expected is not None and actual <= expected
        if op == _op.in_op:
            return actual in expected
    except TypeError:
        return False
    return True


class _Query:
    def __init__(self, source, records=None):
        self._source = source                       # the real store list
        self.records = list(source) if records is None else records

    def filter(self, *args, **kwargs):
        for expr in args:
            self.records = [r for r in self.records if _matches(r, expr)]
        return self

    def first(self):
        return self.records[0] if self.records else None

    def count(self):
        return len(self.records)

    def all(self):
        return list(self.records)

    def update(self, values, synchronize_session=False):
        for record in self.records:
            for key, value in values.items():
                setattr(record, key, value)
        return len(self.records)

    def delete(self, synchronize_session=False):
        count = 0
        for record in list(self.records):
            if record in self._source:
                self._source.remove(record)
                count += 1
        self.records = []
        return count


class _Database:
    def __init__(self, records=None):
        self.records = records or {}

    def query(self, model):
        return _Query(self.records.setdefault(model, []))

    def add(self, value):
        self.records.setdefault(type(value), []).append(value)

    def commit(self):
        pass

    def refresh(self, value):
        pass


def test_identity_endpoint_updates_only_the_authenticated_device():
    device = Device(userId=7, device_uuid="device-id", device_name="Old name", device_type="thoth")
    database = _Database({Device: [device]})
    token_user = mock.Mock(userId=7)
    token_user.get.side_effect = lambda key, default=None: {
        "scopes": ["device"], "device_id": "device-id",
    }.get(key, default)

    with mock.patch(
        "server.endpoints.device_endpoints.get_user_from_token",
        new=mock.AsyncMock(return_value=token_user),
    ):
        result = asyncio.run(update_device_identity(
            "device-id", {"device_name": "Bedroom Thoth"}, "Bearer device-token", database,
        ))

    assert result["device_name"] == "Bedroom Thoth"
    assert device.device_name == "Bedroom Thoth"


def test_identity_endpoint_rejects_a_token_for_another_device():
    device = Device(userId=7, device_uuid="device-id", device_name="Old name", device_type="thoth")
    database = _Database({Device: [device]})
    token_user = mock.Mock(userId=7)
    token_user.get.side_effect = lambda key, default=None: {
        "scopes": ["device"], "device_id": "different-device",
    }.get(key, default)

    with mock.patch(
        "server.endpoints.device_endpoints.get_user_from_token",
        new=mock.AsyncMock(return_value=token_user),
    ):
        try:
            asyncio.run(update_device_identity(
                "device-id", {"device_name": "Bedroom Thoth"}, "Bearer device-token", database,
            ))
        except HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("a token for another device must be rejected")


def test_device_can_start_and_account_can_claim_pairing():
    database = _Database()
    started = asyncio.run(start_device_pairing(DevicePairingStartRequest(
        device_id="physical-device",
        device_name="Bedroom Thoth",
        hardware_info={"hostname": "thoth"},
    ), database, None))
    pairing = database.records[DevicePairing][0]
    # The endpoint stores only hashes; use the returned code by assigning its
    # hash to the in-memory record exactly as a database lookup would.
    assert len(started["code"]) == 8
    user = User(userId=7, username="owner", email="owner@example.com", role=0)

    claimed = asyncio.run(claim_device_pairing(
        DevicePairingClaimRequest(code=started["code"]), current_user=user, db=database,
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


def test_fresh_physical_pairing_moves_an_existing_device_to_the_new_account():
    now = datetime.utcnow()
    pairing = DevicePairing(
        device_uuid="device-id",
        device_name="Re-paired Thoth",
        device_type="thoth",
        hardware_info='{"hostname":"thoth"}',
        code_hash="code-hash",
        secret_hash="secret-hash",
        status="pending",
        expires_at=now + timedelta(minutes=5),
    )
    device = Device(
        deviceId=3,
        userId=6,
        device_uuid="device-id",
        device_name="Old Thoth",
        device_type="thoth",
        online=True,
        last_seen=now,
    )
    database = _Database({
        DevicePairing: [pairing],
        Device: [device],
        DeviceFile: [DeviceFile(device_id=3, user_id=6, filename="radar.bin")],
        DeviceCaptureChunk: [DeviceCaptureChunk(
            device_id=3, user_id=6, minute="20260719_0508", chunk_index=0,
        )],
        DeviceCommand: [DeviceCommand(
            device_id=3, user_id=6, command="capture", payload="{}",
        )],
        FileDeviceUpdate: [FileDeviceUpdate(fileId=4, deviceId=3)],
        DeviceDeployment: [DeviceDeployment(
            deployment_id="deploy-1", device_uuid="device-id", model_id=2,
            user_id=6, payload="{}",
        )],
    })
    new_owner = User(userId=7, username="new-owner", email="new@example.com", role=0)

    with mock.patch("server.endpoints.device_endpoints._pairing_hash", return_value="code-hash"):
        result = asyncio.run(claim_device_pairing(
            DevicePairingClaimRequest(code="PAIRCODE"), current_user=new_owner, db=database,
        ))

    assert result["status"] == "paired"
    assert device.userId == 7
    assert device.device_name == "Re-paired Thoth"
    assert device.online is False
    assert device.last_seen is None
    assert pairing.user_id == 7
    assert not database.records[DeviceFile]
    assert not database.records[DeviceCaptureChunk]
    assert not database.records[DeviceCommand]
    assert not database.records[FileDeviceUpdate]
    assert not database.records[DeviceDeployment]


def test_active_device_requires_its_current_token_to_start_repairing():
    active_device = Device(
        userId=6,
        device_uuid=NORMALIZED_DEVICE_ID,
        device_name="Old Thoth",
        device_type="thoth",
        online=True,
        last_seen=datetime.utcnow(),
    )
    database = _Database({Device: [active_device]})

    try:
        asyncio.run(start_device_pairing(DevicePairingStartRequest(
            device_id="device-id",
            device_name="Re-paired Thoth",
        ), database, None))
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("an existing device must not be re-paired without its token")


def test_active_device_can_start_repairing_with_its_current_token():
    active_device = Device(
        userId=6,
        device_uuid=NORMALIZED_DEVICE_ID,
        device_name="Old Thoth",
        device_type="thoth",
        online=True,
        last_seen=datetime.utcnow(),
    )
    database = _Database({Device: [active_device]})
    token_user = mock.Mock(userId=6)
    token_user.get.side_effect = lambda key, default=None: {
        "scopes": ["device"], "device_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, "device-id")),
    }.get(key, default)

    with mock.patch(
        "server.endpoints.device_endpoints.get_user_from_token",
        new=mock.AsyncMock(return_value=token_user),
    ):
        started = asyncio.run(start_device_pairing(DevicePairingStartRequest(
            device_id="device-id",
            device_name="Re-paired Thoth",
        ), database, "Bearer current-device-token"))

    assert started["status"] == "pending"
    assert len(started["code"]) == 8


def test_stale_device_can_recover_when_its_previous_token_expired():
    stale_device = Device(
        userId=6,
        device_uuid=NORMALIZED_DEVICE_ID,
        device_name="Old Thoth",
        device_type="thoth",
        online=True,
        last_seen=datetime.utcnow() - timedelta(minutes=5),
    )
    database = _Database({Device: [stale_device]})

    started = asyncio.run(start_device_pairing(DevicePairingStartRequest(
        device_id="device-id",
        device_name="Recovered Thoth",
    ), database, None))

    assert started["status"] == "pending"
    assert len(started["code"]) == 8


def test_stale_device_can_recover_with_the_intended_new_account_token():
    stale_device = Device(
        userId=6,
        device_uuid=NORMALIZED_DEVICE_ID,
        device_name="Old Thoth",
        device_type="thoth",
        online=True,
        last_seen=datetime.utcnow() - timedelta(minutes=5),
    )
    database = _Database({Device: [stale_device]})
    new_owner_token = mock.Mock(userId=7)
    new_owner_token.get.side_effect = lambda key, default=None: {
        "scopes": [], "device_id": None,
    }.get(key, default)

    with mock.patch(
        "server.endpoints.device_endpoints.get_user_from_token",
        new=mock.AsyncMock(return_value=new_owner_token),
    ):
        started = asyncio.run(start_device_pairing(DevicePairingStartRequest(
            device_id="device-id",
            device_name="Recovered Thoth",
        ), database, "Bearer new-owner-token"))

    assert started["status"] == "pending"
