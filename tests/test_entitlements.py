from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from server.entitlements import GB, get_entitlements, check_device_limit, normalize_plan


@pytest.mark.parametrize('plan,devices,quota,download,labs', [
    ('free', 1, None, False, False), ('home', 5, 10 * GB, True, False),
    ('research', 10, 100 * GB, True, True),
])
def test_plan_matrix(plan, devices, quota, download, labs):
    user = SimpleNamespace(plan=plan)
    ent = get_entitlements(user)
    assert (ent['device_limit'], ent['storage_bytes'], ent['download_data'], ent['labs']) == (devices, quota, download, labs)
    check_device_limit(user, devices - 1)
    with pytest.raises(HTTPException) as caught:
        check_device_limit(user, devices)
    assert caught.value.status_code == 403


def test_unknown_plan_fails_closed():
    assert normalize_plan('organization') == normalize_plan('pro') == 'free'


def test_free_download_forbidden(api):
    client, _, _ = api
    assert client.get('/api/file/minute/20260922_0000/download').status_code == 403
