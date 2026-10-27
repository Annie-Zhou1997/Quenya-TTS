"""
Quenya grapheme-to-phoneme conversion (Third Age pronunciation, The Lord of the Rings, Appendix E).

    >>> from quenya_g2p import to_ipa
    >>> to_ipa("Elen síla lúmenn' omentielvo")
    'ˈɛlɛn sˈiːla lˈuːmɛnn ɔmɛntiˈɛlvɔ'

The rules, and the choices made where the sources disagree, are documented in quenya_g2p/rules.py.
"""

from quenya_g2p.rules import process_complete_sentence
from quenya_g2p.rules import quenya_to_ipa
from quenya_g2p.rules import quenya_word_to_ipa

to_ipa = process_complete_sentence

__all__ = ["to_ipa", "process_complete_sentence", "quenya_word_to_ipa", "quenya_to_ipa"]
__version__ = "0.1.0"
