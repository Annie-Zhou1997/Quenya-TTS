"""Tests for the file handling of quenya_tts.prepare_corpus (no audio packages needed)."""

import os
import tempfile
import unittest

from quenya_tts.prepare_corpus import clip_name, pair_files, pair_script, read_lines, read_script, write_transcript


class PairFilesTest(unittest.TestCase):

    def test_pairs_recordings_with_texts_of_the_same_name(self):
        with tempfile.TemporaryDirectory() as audio, tempfile.TemporaryDirectory() as texts:
            for name in ("namarie.mp3", "markirya.wav", "notext.mp3", "cover.jpg"):
                open(os.path.join(audio, name), "w").close()
            for name in ("namarie.txt", "markirya.txt", "noaudio.txt"):
                open(os.path.join(texts, name), "w").close()
            pairs = pair_files(audio, texts)
        self.assertEqual([stem for stem, _, _ in pairs], ["markirya", "namarie"])
        self.assertTrue(pairs[1][1].endswith("namarie.mp3"))
        self.assertTrue(pairs[1][2].endswith("namarie.txt"))


class ScriptTest(unittest.TestCase):

    def test_pairs_recordings_named_by_line_id(self):
        with tempfile.TemporaryDirectory() as folder:
            script_path = os.path.join(folder, "script.tsv")
            with open(script_path, "w", encoding="utf8") as f:
                f.write("id\ttext\tsource\tgloss\ns01_001\tAlla!\tbook:1/1.1\t\ns01_002\tMára.\tbook:1/1.2\t\n")
            script = read_script(script_path)
            self.assertEqual(script, {"s01_001": "Alla!", "s01_002": "Mára."})
            audio = os.path.join(folder, "audio")
            os.mkdir(audio)
            for name in ("s01_002.wav", "s01_001.flac", "s09_999.wav", "notes.txt"):
                open(os.path.join(audio, name), "w").close()
            pairs = pair_script(audio, script)
        self.assertEqual([(line_id, text) for line_id, _, text in pairs], [("s01_001", "Alla!"), ("s01_002", "Mára.")])
        self.assertTrue(pairs[0][1].endswith("s01_001.flac"))


class TranscriptTest(unittest.TestCase):

    def test_round_trip_with_blank_lines_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            text = os.path.join(folder, "namarie.txt")
            with open(text, "w", encoding="utf8") as f:
                f.write("Ai! laurië lantar lassi súrinen,\n\n  yéni únótimë ve rámar aldaron!  \n")
            lines = read_lines(text)
            self.assertEqual(lines, ["Ai! laurië lantar lassi súrinen,", "yéni únótimë ve rámar aldaron!"])

            transcript = os.path.join(folder, "transcript.txt")
            write_transcript(transcript, [(clip_name("namarie", n), line) for n, line in enumerate(lines, 1)])
            with open(transcript, encoding="utf8") as f:
                self.assertEqual(f.read().splitlines(),
                                 ["namarie_01.wav|Ai! laurië lantar lassi súrinen,",
                                  "namarie_02.wav|yéni únótimë ve rámar aldaron!"])


if __name__ == "__main__":
    unittest.main()
