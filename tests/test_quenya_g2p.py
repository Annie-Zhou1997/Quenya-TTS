"""
Tests for the Quenya grapheme-to-phoneme rules (Third Age pronunciation,
Appendix E). Run from the repository root with:

    python -m unittest discover tests
"""

import os
import re
import unicodedata
import unittest

from quenya_g2p import process_complete_sentence
from quenya_g2p import quenya_to_ipa
from quenya_g2p import quenya_word_to_ipa

TOUCAN_FEATURES = os.path.join(os.path.dirname(__file__), "..", "toucan", "Preprocessing", "articulatory_features.py")


class StressTest(unittest.TestCase):

    def check(self, cases):
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(quenya_word_to_ipa(word), expected)

    def test_two_syllables_stress_the_first(self):
        self.check({
            "Elen":  "ˈɛlɛn",
            "heru":  "hˈɛru",
            "lassi": "lˈassi",
            "Ainur": "ˈainur",
        })

    def test_appendix_e_examples(self):
        # the stress examples Appendix E gives: isIldur, orOmë, erEssëa, fËanor,
        # ancAlima, elentÁri, andÚnë
        self.check({
            "Isildur":  "isˈildur",
            "Oromë":    "ˈɔrɔmɛ",
            "Eressëa":  "ɛrˈɛssɛa",
            "Fëanor":   "fˈɛanɔr",
            "ancalima": "aŋkˈalima",
            "Elentári": "ɛlɛntˈaːri",
            "andúnë":   "andˈuːnɛ",
        })

    def test_heavy_penultimate_is_stressed(self):
        self.check({
            "Valinórë":   "valinˈoːrɛ",   # long vowel
            "Endorenna":  "ɛndɔrˈɛnna",   # double consonant
            "Elenna":     "ɛlˈɛnna",
            "omentielvo": "ɔmɛntiˈɛlvɔ",  # consonant cluster
            "atarinya":   "atarˈiɲa",     # ny counts as two consonants
            "quetelyë":   "kwɛtˈɛʎɛ",     # ly counts as two consonants
            "Calacirya":  "kalakˈirja",   # ry counts as two consonants
            "Oiolossë":   "ɔiɔlˈɔssɛ",
            "Cuiviénen":  "kuiviˈeːnɛn",
        })

    def test_light_penultimate_moves_stress_back(self):
        self.check({
            "utúlien":      "utˈuːliɛn",
            "Eldalië":      "ɛldˈaliɛ",
            "Ilúvatar":     "ilˈuːvatar",
            "laurië":       "lˈauriɛ",
            "Hyarmendacil": "çarmˈɛndakil",
            "Eruhantalë":   "ɛruhˈantalɛ",
            "hlócë":        "lˈoːkɛ",      # hl counts as one consonant and is l
        })


class SoundTest(unittest.TestCase):

    def check(self, cases):
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(quenya_word_to_ipa(word), expected)

    def test_vowels(self):
        self.check({
            "Namárië": "namˈaːriɛ",   # ë after i is its own syllable
            "Fëanáro": "fɛanˈaːrɔ",
            "yéni":    "jˈeːni",
            "rámar":   "rˈaːmar",
            "nōrë":    "nˈoːrɛ",      # macron marks length too
            "nûmen":   "nˈuːmɛn",     # so does a circumflex
        })

    def test_diphthongs(self):
        self.check({
            "aiya":  "ˈaija",
            "Ainulindalë": "ainulˈindalɛ",
            "leuca": "lˈɛuka",
            "tuilë": "tˈuilɛ",
            "tiuca": "tjˈuːka",  # iu is a rising diphthong in the Third Age, long like yule
            "kiuka": "kjˈuːka",  # Tolkien's own [kyūka], PE19 p. 107: k not fronted before this y
        })

    def test_h_combinations(self):
        self.check({
            "hwesta":  "ʍˈɛsta",
            "hyarmen": "çˈarmɛn",
            "hlócë":   "lˈoːkɛ",    # hl is l in the Third Age
            "hrívë":   "rˈiːvɛ",    # and hr is r
            "ohtar":   "ˈɔxtar",
            "tehtar":  "tˈɛçtar",
            "ehtyar":  "ˈɛçcar",    # also before ty
            "ahtya":   "ˈaçca",     # before ty the h is fronted after any vowel (PE19 p. 84; made-up word)
            "mahalma": "mahˈalma",  # plain h between vowels
        })

    def test_nasals(self):
        self.check({
            "ingor":    "ˈiŋɡɔr",
            "enquë":    "ˈɛŋkwɛ",
            "Ñoldor":   "nˈɔldɔr",   # initial ñ is n in the Third Age
            "ngwalmë":  "nwˈalmɛ",
            "Angamando": "aŋɡamˈandɔ",
        })

    def test_palatals_and_w(self):
        self.check({
            "tyelpë":   "cˈɛlpɛ",
            "Elwë":     "ˈɛlwɛ",      # w stays after a consonant (PE17 p. 129)
            "Aiwendil": "aiwˈɛndil",  # and after ai, oi (PE19 p. 72)
            "aiwë":     "ˈaiwɛ",
            "wingë":    "vˈiŋɡɛ",     # initial w is v in the Third Age
            "awa":      "ˈava",       # and so is w between vowels (PE17 pp. 76, 129; PE19 p. 72)
            "tawar":    "tˈavar",     # older spelling of tavar
            "wanwa":    "vˈanwa",     # initial v, nw kept
        })

    def test_th_x_q_and_z(self):
        self.check({
            "þúlë":      "sˈuːlɛ",
            "Helcaraxë": "hɛlkarˈaksɛ",
            "Qenya":     "kwˈɛɲa",   # Tolkien's older spelling of qu
            "áze":       "ˈaːrɛ",    # archaic z is read as the later r
        })

    def test_old_entry_point(self):
        self.assertEqual(quenya_to_ipa("Elen"), "ˈɛlɛn")


class SentenceTest(unittest.TestCase):

    def test_namarie_line(self):
        self.assertEqual(
            process_complete_sentence("Ai! laurië lantar lassi súrinen, yéni únótimë ve rámar aldaron!"),
            "ˈai! lˈauriɛ lˈantar lˈassi sˈuːrinɛn, jˈeːni uːnˈoːtimɛ vɛ rˈaːmar ˈaldarɔn!")

    def test_hyphen_apostrophe_and_function_words(self):
        self.assertEqual(
            process_complete_sentence("Sinome maruvan ar Hildinyar, tenn’ Ambar-metta!"),
            "sˈinɔmɛ mˈaruvan ar hildˈiɲar, tˈɛnn ˈambar mˈɛtta!")
        self.assertEqual(
            process_complete_sentence("Aiya i Eldar mi Valinor"),
            "ˈaija i ˈɛldar mi vˈalinɔr")
        self.assertEqual(process_complete_sentence("ohta·caro"), "ˈɔxta kˈarɔ")

    def test_pauses_for_quotes_dashes_and_line_breaks(self):
        self.assertEqual(
            process_complete_sentence("«Aiya Eärendil» — elenion ancalima\n(quetë Frodo)"),
            "ˈaija ɛarˈɛndil, ɛlˈɛniɔn aŋkˈalima, kwˈɛtɛ frˈɔdɔ")

    def test_stray_tokens_and_empty_input(self):
        self.assertEqual(process_complete_sentence("Elen q síla"), "ˈɛlɛn sˈiːla")
        self.assertEqual(process_complete_sentence("ß 12 q"), "")
        self.assertEqual(process_complete_sentence(""), "")
        self.assertEqual(process_complete_sentence("Namárië, "), "namˈaːriɛ")

    def test_decomposed_unicode_input(self):
        text = "Namárië yéni"
        self.assertEqual(
            process_complete_sentence(unicodedata.normalize("NFD", text)),
            process_complete_sentence(text))

    def test_every_word_has_one_stress(self):
        text = "Elen síla lúmenn omentielvo. Ma quetelyë Quenya?"
        for word in re.findall(r"[^\s,.!?]+", process_complete_sentence(text)):
            with self.subTest(word=word):
                self.assertEqual(word.count("ˈ"), 1)

    @unittest.skipUnless(os.path.exists(TOUCAN_FEATURES), "IMS-Toucan submodule not checked out")
    def test_output_only_uses_known_phones(self):
        # every phone must be one IMS-Toucan has articulatory features for
        with open(TOUCAN_FEATURES, encoding="utf8") as f:
            known = set(re.findall(r"^        '([^']+)': \{", f.read(), re.M))
        known |= {"ː", "ˈ", ",", ";", ":"}
        text = ("Ai! laurië lantar lassi súrinen, yéni únótimë ve rámar aldaron! "
                "hwesta hyarmen hlócë hrívë hróa tehtar ohtar ehtyar ingor enquë Ñoldor ngwalmë "
                "tyelpë Elwë Aiwendil aiwë wingë awa tawar wanwa þúlë Helcaraxë tiuca Qenya áze Calacirya Hyarmendacil")
        self.assertEqual(set(process_complete_sentence(text)) - known, set())


if __name__ == "__main__":
    unittest.main()
