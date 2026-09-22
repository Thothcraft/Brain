from server.db import File, Device, DeviceFile, TrainedModel
from server.entitlements import GB
from server.storage import user_storage_bytes, storage_status


def test_storage_counts_only_owned_cloud_objects(api):
    _, db, user = api
    db.add(Device(deviceId=1, userId=user.userId, device_uuid='pi', device_name='Pi'))
    db.add_all([
        File(userId=1, filename='own', size=10),
        File(userId=2, filename='other', size=1000),
        DeviceFile(device_id=1, user_id=1, filename='cloud', size=20, on_cloud=True),
        DeviceFile(device_id=1, user_id=1, filename='local', size=999, on_cloud=False),
        TrainedModel(user_id=1, name='model', size_bytes=30),
    ])
    db.commit()
    assert user_storage_bytes(db, user) == 60
    assert storage_status(db, user)['minute_retention'] == 400


def test_upload_rejected_at_quota(api):
    client, db, user = api
    user.plan = 'home'
    db.add(File(userId=user.userId, filename='full', size=10 * GB))
    db.commit()
    response = client.post('/api/file/upload', json={'filename': 'new.txt', 'content': 'x', 'is_base64': False})
    assert response.status_code == 413, response.text
    assert db.query(File).count() == 1
