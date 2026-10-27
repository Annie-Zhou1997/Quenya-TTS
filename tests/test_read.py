import unittest

import numpy as np

from quenya_tts.read import silence_lead


class SilenceLeadTest(unittest.TestCase):

    def test_mutes_lead_and_fades_in(self):
        wave = np.ones(1000, dtype=np.float32)
        out = silence_lead(wave, sr=1000, lead_frames=2, total_frames=10, fade_seconds=0.05)
        self.assertTrue(np.all(out[:150] == 0))
        self.assertTrue(np.all(np.diff(out[150:200]) > 0))
        self.assertTrue(np.all(out[200:] == 1))
        self.assertTrue(np.all(wave == 1))


if __name__ == "__main__":
    unittest.main()
