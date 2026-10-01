import base64
import gzip
import hashlib
import json
import os
import unittest

from result_transport import stream_files


class StreamTransportTests(unittest.TestCase):
    def test_glb_only_stream_is_lossless_and_bounded(self):
        glb = os.urandom(40 * 1024 * 1024 + 12345)
        digest = hashlib.sha256()
        next_index = 0
        events = list(stream_files({"glb": glb}))
        self.assertEqual(events[0]["type"], "manifest")
        self.assertEqual(list(events[0]["files"]), ["glb"])
        self.assertEqual(events[0]["files"]["glb"]["sha256"], hashlib.sha256(glb).hexdigest())
        self.assertEqual(events[-1]["type"], "complete")
        for event in events:
            self.assertLess(len(json.dumps({"output": event}).encode()), 1024 * 1024)
            if event["type"] == "chunk":
                self.assertEqual(event["index"], next_index)
                next_index += 1
                digest.update(gzip.decompress(base64.b64decode(event["data"])))
        self.assertEqual(next_index, events[0]["files"]["glb"]["chunks"])
        self.assertEqual(digest.digest(), hashlib.sha256(glb).digest())

    def test_empty_files_are_dropped_and_all_empty_is_an_error(self):
        events = list(stream_files({"ply": b"", "glb": b"glTF"}))
        self.assertEqual(list(events[0]["files"]), ["glb"])
        with self.assertRaisesRegex(ValueError, "ningún archivo"):
            list(stream_files({"glb": b""}))

    def test_rejects_kinds_the_receiver_does_not_know(self):
        with self.assertRaisesRegex(ValueError, "no admitido"):
            list(stream_files({"obj": b"x"}))


if __name__ == "__main__":
    unittest.main()
