"""Tests for the parts of quenya_tts.watermark that need no model."""

import unittest

import numpy as np

from quenya_tts import watermark


class WatermarkTest(unittest.TestCase):

    def test_message_round_trips_through_bits(self):
        bits = watermark.to_bits(watermark.MESSAGE)
        self.assertEqual(len(bits), 16)
        self.assertEqual(watermark.from_bits(bits), 0x5154)

    def test_fit_length_cuts_and_pads(self):
        self.assertEqual(watermark.fit_length(np.ones(5), 3).tolist(), [1, 1, 1])
        self.assertEqual(watermark.fit_length(np.ones(2), 4).tolist(), [1, 1, 0, 0])

    def test_generated_files_are_labelled(self):
        import os
        import tempfile
        try:
            import soundfile
        except ImportError:
            self.skipTest("soundfile is not installed")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "x.wav")
            watermark.write_generated(path, np.zeros(1600, dtype=np.float32), 16000)
            with soundfile.SoundFile(path) as f:
                self.assertIn("AI-generated", f.comment)
                self.assertIn("Quenya-TTS", f.software)


if __name__ == "__main__":
    unittest.main()
