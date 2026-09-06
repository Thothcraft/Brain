import io
import zipfile
from types import SimpleNamespace

from server.endpoints import file_endpoints


def test_stream_zip_yields_a_valid_archive_and_reuses_storage_client(monkeypatch):
    records = {
        1: SimpleNamespace(filename="first.bin"),
        2: SimpleNamespace(filename="second.csv"),
    }
    contents = {
        "first.bin": b"\x00\x01radar",
        "second.csv": b"timestamp,value\n1,2\n",
    }
    storage_clients = []

    def fake_file_content(record, storage_client=None):
        storage_clients.append(storage_client)
        return contents[record.filename]

    monkeypatch.setattr(file_endpoints, "_file_content", fake_file_content)
    payload = b"".join(file_endpoints._stream_zip([
        (1, "20260824_1200/first.bin"),
        (2, "20260824_1201/second.csv"),
    ], record_loader=records.get))

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.namelist() == [
            "20260824_1200/first.bin",
            "20260824_1201/second.csv",
        ]
        assert archive.read("20260824_1200/first.bin") == contents["first.bin"]
        assert archive.read("20260824_1201/second.csv") == contents["second.csv"]

    assert storage_clients[0] is storage_clients[1]
