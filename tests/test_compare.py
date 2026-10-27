"""Tests for the text and alignment helpers of quenya_tts.compare (they need no audio packages)."""

import unittest

from quenya_tts.compare import align
from quenya_tts.compare import cut_by_phones
from quenya_tts.compare import estimate_lines
from quenya_tts.compare import find_lines
from quenya_tts.compare import ipa_to_phones
from quenya_tts.compare import phone_error_rate
from quenya_tts.compare import snap_to_quiet


class PhonesTest(unittest.TestCase):

    def test_drops_stress_pauses_and_length(self):
        self.assertEqual(ipa_to_phones("ˈɛlɛn sˈiːla, lˈuːmɛnn"),
                         ["ɛ", "l", "ɛ", "n", "s", "i", "l", "a", "l", "u", "m", "ɛ", "n", "n"])

    def test_keeps_length_on_request(self):
        self.assertEqual(ipa_to_phones("siːla", keep_length=True), ["s", "iː", "l", "a"])

    def test_unifies_spelling_variants(self):
        self.assertEqual(ipa_to_phones("ɾ g ɹ"), ["r", "ɡ", "r"])

    def test_diacritics_stay_with_their_phone(self):
        self.assertEqual(ipa_to_phones("tʰa çe"), ["tʰ", "a", "ç", "e"])


class AlignTest(unittest.TestCase):

    def test_identical(self):
        self.assertEqual(phone_error_rate(list("abc"), list("abc")), (0.0, []))

    def test_substitution_insertion_deletion(self):
        distance, pairs = align(list("abcd"), list("axcde"))
        self.assertEqual(distance, 2)
        self.assertIn(("b", "x"), pairs)
        self.assertIn((None, "e"), pairs)
        per, diffs = phone_error_rate(list("ab"), list("a"))
        self.assertEqual((per, diffs), (0.5, [("b", None)]))


class FindLinesTest(unittest.TestCase):

    def test_cuts_at_longest_pauses(self):
        # speech stretches with short pauses inside lines and long pauses between them
        intervals = [(0, 10), (12, 20), (40, 50), (52, 60), (90, 100)]
        self.assertEqual(find_lines(intervals, 3), [(0, 20), (40, 60), (90, 100)])

    def test_too_few_stretches(self):
        with self.assertRaises(ValueError):
            find_lines([(0, 10)], 2)


class EstimateLinesTest(unittest.TestCase):

    def test_snaps_to_a_nearby_pause(self):
        # two stretches of speech, three lines of equal length: one cut lands in the pause, one is estimated
        spans = estimate_lines([(0, 100), (110, 300)], [1, 1, 1])
        self.assertEqual(len(spans), 3)
        self.assertEqual(spans[0], (0, 105))
        self.assertEqual(spans[-1][1], 300)
        self.assertTrue(all(start < end for start, end in spans))

    def test_follows_line_lengths_without_pauses(self):
        self.assertEqual(estimate_lines([(0, 300)], [1, 2]), [(0, 100), (100, 300)])


class CutByPhonesTest(unittest.TestCase):

    def test_cuts_between_lines_even_without_pauses(self):
        lines = [list("abc"), list("def"), list("gh")]
        # heard at 1 s per phone with no pauses, one phone misheard and one extra
        heard = [(p, float(t)) for t, p in enumerate("abxdefzgh")]
        spans = cut_by_phones(lines, heard, 10.0)
        self.assertEqual(len(spans), 3)
        self.assertAlmostEqual(spans[0][1], 2.5)  # between c (heard as x at 2 s) and d at 3 s
        self.assertAlmostEqual(spans[1][1], 6.0)  # between f at 5 s and g at 7 s, past the extra z
        self.assertEqual(spans[0][0], 0.0)
        self.assertAlmostEqual(spans[2][1], 8.5)

    def test_line_not_heard_is_interpolated(self):
        spans = cut_by_phones([list("ab"), list("xyz"), list("cd")], [("a", 0.0), ("b", 1.0), ("c", 5.0), ("d", 6.0)], 8.0)
        self.assertTrue(spans[0][1] <= spans[1][0] + 1e-9 or abs(spans[0][1] - spans[1][0]) < 1e-9)
        self.assertTrue(1.0 < spans[1][1] < 5.0)


class SnapToQuietTest(unittest.TestCase):

    def test_moves_cut_into_nearby_gap(self):
        loudness = [1.0] * 100
        loudness[40:45] = [0.01] * 5  # a pause at 0.40-0.44 s, frames of 10 ms
        self.assertAlmostEqual(snap_to_quiet(0.62, loudness, 0.01, window=0.3), 0.44)

    def test_keeps_cut_without_gap_in_reach(self):
        loudness = [1.0] * 100
        loudness[10] = 0.0
        self.assertAlmostEqual(snap_to_quiet(0.62, loudness, 0.01, window=0.2), 0.62)


if __name__ == "__main__":
    unittest.main()
