"""Tests for the release checks in quenya_tts.publish (no models needed)."""

import json
import os
import tempfile
import unittest

import numpy as np

from quenya_tts import publish


class PickVoiceTest(unittest.TestCase):

    def test_takes_the_first_voice_unlike_every_volunteer(self):
        volunteers = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        candidates = [np.array([0.9, 0.1, 0.0]),   # sounds like the first volunteer
                      np.array([0.1, 0.0, 1.0])]   # like nobody
        index, highest = publish.pick_voice(candidates, volunteers, threshold=0.25)
        self.assertEqual(index, 1)
        self.assertLess(highest, 0.25)

    def test_refuses_when_every_voice_is_too_close(self):
        with self.assertRaisesRegex(ValueError, "closest: 1.00"):
            publish.pick_voice([np.array([1.0, 0.1])], [np.array([1.0, 0.0])], threshold=0.25)


class ReleaseProblemTest(unittest.TestCase):

    def test_only_models_trained_on_public_corpora_are_released(self):
        self.assertIn("no training.json", publish.release_problem(None))
        self.assertIn("Atanquesta", publish.release_problem({"corpora": ["Atanquesta", "Volunteers"], "public": False}))
        self.assertIsNone(publish.release_problem({"corpora": ["Volunteers"], "public": True}))

    def test_reads_training_json_next_to_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(publish.training_info(os.path.join(folder, "best.pt")))
            with open(os.path.join(folder, "training.json"), "w") as f:
                json.dump({"corpora": ["Volunteers"], "public": True}, f)
            self.assertEqual(publish.training_info(os.path.join(folder, "best.pt"))["corpora"], ["Volunteers"])


class VolunteerFilesTest(unittest.TestCase):

    def test_groups_clips_by_volunteer_and_reads_credits(self):
        with tempfile.TemporaryDirectory() as folder:
            for volunteer, recording, line in [("aa", "r1", "s01_001"), ("aa", "r2", "s02_001"), ("bb", "r1", "s01_001")]:
                os.makedirs(os.path.join(folder, volunteer, recording), exist_ok=True)
                open(os.path.join(folder, volunteer, recording, f"{line}.wav"), "wb").close()
            with open(os.path.join(folder, "volunteers.tsv"), "w") as f:
                f.write("folder\tspeaker\tcredit_name\nx\tva\tAna\ny\tva\tAna\nz\tvb\t\n")
            clips = publish.volunteer_clips(folder)
            self.assertEqual(sorted(clips), ["aa", "bb"])
            self.assertEqual(len(clips["aa"]), 2)
            self.assertEqual(publish.credit_names(folder), ["Ana"])


if __name__ == "__main__":
    unittest.main()
