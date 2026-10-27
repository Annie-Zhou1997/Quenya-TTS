"""Tests for quenya_tts.volunteers with an in-memory, encrypted bucket (no storage account needed)."""

import base64
import datetime
import json
import os
import tempfile
import unittest

from quenya_tts import volunteers

SITE = {"minimum_training_minutes": 60, "training_wait_days": 30}
UTC = datetime.timezone.utc


class FakeClient:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, objects):
        self.objects = objects  # key -> (bytes, metadata)

    def get_paginator(self, _):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": key, "Size": len(data)} for key, (data, _) in sorted(objects.items())
                                    if key.startswith(Prefix)]}
        return Paginator()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey(Key)

        class Body:
            def __init__(self, data):
                self.data = data

            def read(self):
                return self.data
        return {"Body": Body(self.objects[Key][0])}

    def head_object(self, Bucket, Key):
        return {"Metadata": self.objects[Key][1]}


KEY = bytes([9] * 32)


def bucket_with(objects):
    """Stores the objects encrypted, the way the site stores them."""
    def sealed_meta(meta):
        return {"sealed": base64.b64encode(volunteers.seal(KEY, json.dumps(meta).encode())).decode()}
    encoded = {key: (volunteers.seal(KEY, json.dumps(value).encode() if isinstance(value, dict) else value),
                     sealed_meta(meta))
               for key, (value, meta) in objects.items()}
    return volunteers.Bucket("test", FakeClient(encoded), KEY)


class EncryptionTest(unittest.TestCase):

    def test_reads_what_the_site_encrypted(self):
        # made by quenya-voices src/storage.js seal(), with the key of 32 bytes of 9
        body = base64.b64decode("UVZFMSP5bceZ1bR0xnCsPnHpPrrty/ELcz6magHDi2mdgSM0SumXN0mt")
        meta = base64.b64decode("UVZFMSfqug6N9HQQCy+mMjXT/Vjs6HXIMQgCQmQbFgZ3G47FCMdDaLQA6/HixnhE6lpgtWq2+fIrrp/3wmiMkQ==")
        self.assertEqual(volunteers.unseal(KEY, body).decode(), "Elen síla")
        self.assertEqual(json.loads(volunteers.unseal(KEY, meta)), {"seconds": "3.25", "name": "Ána"})

    def test_wrong_key_or_plain_data_is_refused(self):
        sealed = volunteers.seal(KEY, b"data")
        with self.assertRaises(Exception):
            volunteers.unseal(bytes(32), sealed)
        with self.assertRaises(ValueError):
            volunteers.unseal(KEY, b"RIFF....plain wav")

    def test_key_must_be_32_bytes(self):
        self.assertEqual(volunteers.storage_key(base64.b64encode(KEY).decode()), KEY)
        with self.assertRaises(ValueError):
            volunteers.storage_key("")

    def test_sizes_are_those_of_the_decrypted_objects(self):
        bucket = bucket_with({"a/b": (b"12345", {})})
        self.assertEqual(bucket.keys("a/"), [("a/b", 5)])


VOLUNTEER = "0123456789abcdef"
RECORDING = "rabc123def"


def accepted_site(extra=None):
    objects = {
        f"accepted/{VOLUNTEER}/{RECORDING}": (b"", {"seconds": "4.0"}),
        f"volunteers/{VOLUNTEER}/consent.json": ({"licence": "training-only", "credit_name": "Ana",
                                                 "consent_version": "2026-10-18"}, {}),
        f"recordings/{VOLUNTEER}/{RECORDING}/recording.json": ({
            "id": RECORDING, "session": 1, "status": "accepted", "created_at": "2026-10-18T10:00:00Z",
            "review": {"s01_001": {"status": "good"}, "s01_002": {"status": "good"}}}, {}),
        f"recordings/{VOLUNTEER}/{RECORDING}/s01_001.wav": (b"RIFF-one", {}),
        f"recordings/{VOLUNTEER}/{RECORDING}/s01_002.wav": (b"RIFF-two", {}),
        f"recordings/{VOLUNTEER}/{RECORDING}/s01_003.wav": (b"RIFF-not-reviewed", {}),
    }
    objects.update(extra or {})
    return bucket_with(objects)


class TrainingGateTest(unittest.TestCase):

    def test_waits_for_the_minimum_and_then_the_waiting_period(self):
        now = datetime.datetime(2026, 12, 1, tzinfo=UTC)
        self.assertIn("only 59.0 of the 60", volunteers.training_gate(59 * 60, None, now, SITE))
        self.assertIn("has not recorded", volunteers.training_gate(3600, None, now, SITE))
        milestones = {"minimum_reached_at": "2026-11-15T00:00:00Z"}
        self.assertIn("2026-12-15", volunteers.training_gate(3600, milestones, now, SITE))
        self.assertIsNone(volunteers.training_gate(3600, milestones, datetime.datetime(2026, 12, 16, tzinfo=UTC), SITE))

    def test_uses_the_rules_the_site_saved_with_the_milestone(self):
        now = datetime.datetime(2026, 12, 1, tzinfo=UTC)
        milestones = {"minimum_reached_at": "2026-11-15T00:00:00Z", "minimum_minutes": 120, "training_wait_days": 10}
        self.assertIn("of the 120 minutes", volunteers.training_gate(3600, milestones, now))
        self.assertIsNone(volunteers.training_gate(120 * 60, milestones, now))

    def test_counts_accepted_seconds_from_the_markers(self):
        self.assertEqual(volunteers.accepted_seconds(accepted_site()), {VOLUNTEER: 4.0})


class DownloadTest(unittest.TestCase):

    def test_downloads_only_good_lines_of_accepted_recordings(self):
        with tempfile.TemporaryDirectory() as out:
            result = volunteers.download(accepted_site(), out)
            folder = os.path.join(out, VOLUNTEER, RECORDING)
            self.assertEqual(sorted(os.listdir(folder)), ["recording.json", "s01_001.wav", "s01_002.wav"])
            self.assertEqual(result[0][1], f"v01234567_{RECORDING}")
            with open(os.path.join(out, "volunteers.tsv"), encoding="utf8") as f:
                rows = [line.rstrip("\n").split("\t") for line in f]
            self.assertEqual(rows[1], [f"{VOLUNTEER}/{RECORDING}", f"v01234567_{RECORDING}", "Ana", "training-only",
                                       "2026-10-18", "1", "2"])

    def test_erasure_deletes_local_copies_and_skips_the_volunteer(self):
        with tempfile.TemporaryDirectory() as out:
            volunteers.download(accepted_site(), out)
            erased = bucket_with({
                f"accepted/{VOLUNTEER}/{RECORDING}": (b"", {"seconds": "4.0"}),  # stale marker
                f"erasures/{VOLUNTEER}.json": ({"volunteer": VOLUNTEER, "erased_at": "2026-10-19T00:00:00Z"}, {}),
            })
            result = volunteers.download(erased, out)
            self.assertEqual(result, [])
            self.assertFalse(os.path.exists(os.path.join(out, VOLUNTEER)))


if __name__ == "__main__":
    unittest.main()
