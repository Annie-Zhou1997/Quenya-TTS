"""Tests for the helpers of recording/build_script.py (no Eldamo data needed)."""

import importlib.util
import os
import tempfile
import unittest

_PATH = os.path.join(os.path.dirname(__file__), "..", "recording", "build_script.py")
_SPEC = importlib.util.spec_from_file_location("build_script", _PATH)
build_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_script)


class SpellingTest(unittest.TestCase):

    def test_clean_drops_homonym_numbers_and_optional_letters(self):
        self.assertEqual(build_script.clean("núla¹"), "núla")
        self.assertEqual(build_script.clean("(h)róna"), "róna")

    def test_readable_rejects_archaic_spellings_and_editorial_marks(self):
        self.assertTrue(build_script.readable("Ma quetilyë Quenya?"))
        self.assertTrue(build_script.readable("Minë. — Mára sinyë, Nerdanel!"))
        self.assertFalse(build_script.readable("kesta"))
        self.assertFalse(build_script.readable("[?it/itē] kestallen"))
        self.assertFalse(build_script.readable("Se - hye - hye -."))

    def test_verbs_are_read_as_first_person_aorist(self):
        self.assertEqual(build_script.verb_form("valta-"), "valtan")
        self.assertEqual(build_script.verb_form("tir-"), "tirin")


class StressTest(unittest.TestCase):

    def test_counts_syllables_from_the_end(self):
        self.assertEqual(build_script.stress_patterns("naˈmaːriɛ ˈaija"), ["stress:4/3", "stress:2/2"])


class BookLinesTest(unittest.TestCase):

    def test_reads_turns_and_list_items_without_speakers(self):
        with tempfile.TemporaryDirectory() as audio:
            os.makedirs(os.path.join(audio, "parande_1"))
            with open(os.path.join(audio, "parande_1", "SCRIPT.md"), "w", encoding="utf8") as f:
                f.write("# Script\n\n> not in a track\n\n## 1.1 – greetings\n\n> Alla! … Aiya!\n\n"
                        "## 1.2 – dialogue\n\n> **Elenwë:** Alla, Nerdanel!\n> **Nerdanel:** Alla!\n\n"
                        "## 1.9 – sounds\n\n> á: mára, málo · qu: Quenya\n")
            lines = build_script.book_lines(audio)
        self.assertEqual([text for text, _, _ in lines], ["Alla!", "Aiya!", "Alla, Nerdanel!", "mára, málo", "Quenya"])
        self.assertEqual(lines[2][1], "book:1/1.2")


class SessionTest(unittest.TestCase):

    def test_reads_one_session_in_order(self):
        from quenya_tts.record import session_lines
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "script.tsv")
            with open(path, "w", encoding="utf8") as f:
                f.write("id\ttext\tsource\tgloss\ns01_001\tAlla!\tbook\t\ns01_002\tAiya!\tbook\t\n"
                        "s02_001\tMára.\tbook\tgood\n")
            self.assertEqual(session_lines(path, 1), [("s01_001", "Alla!", ""), ("s01_002", "Aiya!", "")])
            self.assertEqual(session_lines(path, 2), [("s02_001", "Mára.", "good")])


class TakeCheckTest(unittest.TestCase):

    def test_drops_startup_zeros_and_flags_cut_takes(self):
        import numpy as np
        from quenya_tts.record import level_warning, strip_startup
        sr = 1000
        voice = 0.5 * np.sin(np.linspace(0.1, 200, sr))
        quiet = np.zeros(sr // 2) + 0.001
        fading = np.concatenate([quiet, voice, quiet])
        self.assertEqual(len(strip_startup(np.concatenate([np.zeros(300), fading]))), len(fading))
        self.assertIsNone(level_warning(fading, sr))
        self.assertIn("very end", level_warning(np.concatenate([quiet, voice]), sr))
        self.assertIn("very start", level_warning(np.concatenate([voice, quiet]), sr))
        self.assertIn("clipped", level_warning(np.ones(sr), sr))


if __name__ == "__main__":
    unittest.main()
