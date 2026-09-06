import io
import json
import unittest

import numpy as np

from server.utils.capture_container import camera_frame, csi_payload, metadata


def encoded_json(value):
    return np.frombuffer(json.dumps(value).encode("utf-8"), dtype=np.uint8)


class CaptureContainerReaderTests(unittest.TestCase):
    def setUp(self):
        output = io.BytesIO()
        first = b"CSI_DATA,x,[3 4]"
        second = b"CSI_DATA,x,[5 12]"
        camera = b"\xff\xd8image\xff\xd9"
        np.savez(
            output,
            metadata_json=encoded_json({
                "schema": "thoth-capture-npz/v1",
                "seconds": [{"second_index": 0, "camera_frames": 1, "csi_samples": 2}],
            }),
            camera_present=np.asarray([1], dtype=np.uint8),
            camera_jpeg_bytes=np.frombuffer(camera, dtype=np.uint8),
            camera_jpeg_offsets=np.asarray([0, len(camera)], dtype=np.int64),
            csi_sample_second_index=np.asarray([0, 0], dtype=np.int16),
            csi_sample_receiver_index=np.asarray([0, 1], dtype=np.int16),
            csi_sample_unix_ns=np.asarray([10, 20], dtype=np.int64),
            csi_sample_bytes=np.frombuffer(first + second, dtype=np.uint8),
            csi_sample_offsets=np.asarray([0, len(first), len(first) + len(second)], dtype=np.int64),
        )
        self.content = output.getvalue()

    def test_reads_metadata_without_pickle(self):
        self.assertEqual(metadata(self.content)["schema"], "thoth-capture-npz/v1")

    def test_reads_camera_and_bounded_csi(self):
        self.assertEqual(camera_frame(self.content, 0), b"\xff\xd8image\xff\xd9")
        self.assertIsNone(camera_frame(self.content, 1))
        samples = csi_payload(self.content, second_index=0, limit=1)
        self.assertEqual(samples["count"], 1)
        self.assertEqual(samples["samples"][0]["raw_csi_line"], "CSI_DATA,x,[3 4]")


if __name__ == "__main__":
    unittest.main()
