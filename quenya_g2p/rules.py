"""
Quenya grapheme-to-phoneme conversion.

Target pronunciation: Third Age Quenya as described in Appendix E of
The Lord of the Rings. The rules are Quenya-only: Sindarin names in mixed
text (Elbereth, Thorondor, Lúthien) get the Quenya th -> [s] rule, although
Sindarin th is [θ], so they should not be used to judge the output.

- Vowels: short a e i o u are [a ɛ i ɔ u]; long á é í ó ú are [aː eː iː oː uː]
  (Appendix E: long é and ó are tenser and closer than the short vowels).
  Circumflex and macron also mark long vowels. A dieresis (ë, ä, ...) only
  marks a vowel that is pronounced on its own; it never ends a diphthong.
- Diphthongs: ai, au, eu, oi, ui (falling), and iu, which Appendix E says was
  usually a rising diphthong in the Third Age, "as yu in English yule": [juː],
  with the length on the u as in Tolkien's own transcription of kiuka as
  [kyūka] (PE19 p. 107, which also says that ui stayed falling in Third Age
  speech and that the k before this [j] was not fronted to ty).
- c and k are [k], qu (and Tolkien's older spelling q) is [kw], x is [ks],
  g is [ɡ], y is [j], r is trilled. A stray j is read like y.
- h is [h], but [ç] in hy, [ʍ] in hw, and before t it is [ç] after e/i and
  [x] elsewhere (ht as in German echt, acht); before ty it is always [ç],
  because "in the sequence hty h was fronted" (PE19 p. 84).
- hl and hr are [l] and [r]: Appendix E says hl "was in the Third Age usually
  pronounced as l" and refers hr to the same note ("Cf. L"); Tolkien's own
  account is that "among the Noldor hr, hl became voiced to r, l before the
  Exile, and the use of r, l in these cases was normal in TQ, as spoken, tho'
  the spelling was usually maintained" (PE19 p. 79).
- th and þ are [s] (the Exiles' pronunciation of þ: Appendix E, Isil; PE19
  p. 71: "In the actual use of the TQ as a spoken language s for þ remained
  characteristic of the Ñoldor").
- z, which occurs only in archaic or Vanyarin spellings, is read as [r], the
  Third Age (Noldorin) development (áze = áre; PE19 p. 73: the rhotacism "was
  nonetheless a normal feature of Exilic TQ").
- ty, ny, ly are the palatals [c ɲ ʎ]; other consonant + y is consonant + [j].
  Appendix E's "t in English tune" fits either [c] or [tj]; Eldamo says the
  choice barely matters.
- n is [ŋ] before c/k/qu/g; ng is [ŋɡ], final ng is [ŋ]; ñ is [ŋ]. Initial ñ and
  ng are [n] (Third Age: Ñoldor, ngwalmë are said Noldor, nwalmë; PE19
  pp. 76-77: the Exiles' everyday pronunciation and spelling had n, and ñ
  survived as a loremasters' reading pronunciation).
- w is [v] at the start of a word (Appendix E: vilya, "older wilya") and
  between vowels (awa, tawar -> ava, tavar); it stays [w] after a consonant
  (Elwë, nwalmë, hw, qu, ngw) and after the diphthongs ai, oi (Aiwendil,
  aiwë). This is the Third Age (Exilic, Tarquesta) change of w to v as
  Tolkien states it three times: PE17 p. 129 (Exilic Quenya outline,
  1964-65), among the principal Exilic changes, "the change of w to v
  (except in nw, tw, hw, kw, ngw)"; PE17 p. 76 (on Namárië, 1966), "w > v
  (ve, as, like) except after consonants"; PE19 p. 72 (Outline of Phonology,
  c. 1951), w became v "where it stood initially or remained between
  vowels", while "after s (z), n, r, l, and the diphthongs ai, oi however w
  remained as such"; also PE17 p. 154 ("Q b-, w- > v-"), PE17 p. 130 (the
  pronoun we merged with ve by this change) and PE22 p. 44 (1940s: v and w
  merged except after a consonant). Appendix E gives only the initial case
  because Third Age spelling already writes v between vowels (ve, avánier,
  Valimar), so the rule changes no attested Third Age word; it matters for
  Tolkien's older spellings with w (awa, wanwa in PE17) and for learner
  input. Intervocalic w was read as written until 2026-10-26.
- Double consonants are written as two phones (ll -> ll).

Stress: words of two syllables are stressed on the first. Longer words are
stressed on the second-to-last syllable when it is heavy (long vowel,
diphthong, or followed by two or more consonants), otherwise on the
third-to-last. Appendix E does not say how the digraphs count. Tolkien does,
in the Outline of Phonetic Development (PE19 p. 60): "A long syllable in PQ,
TQ is a syllable that contains (a) a long vowel, (b) a diphthong, (c) a vowel
followed by any two (or more) consonants, including ty, hy, kw, hw"; the
earlier draft of the same sentence adds "kw (qu)". So qu, x, ng and any
consonant + y or w (ty ny ly ry hy hw) count as two consonants (as Eldamo
also does), while hl, hr and th count as one. Tolkien's own 1967 stress marks
híruválye and ṑmáryo (The Road Goes Ever On, pp. 58-61) show the suffixes
-lye and -rya making the syllable before them heavy. Known divergence, not
followed: in a 1940s note (PE23 pp. 83-84) the ending -tye "was 'short' and
the vowel before it treated as such", stressed kàritye, nàkitye, where these
rules give ka-RI-tye, na-KI-tye; the same exception appears in the earliest,
rejected draft of the PE19 passage (p. 65: "The inflexional pronouns -tye,
-qe are short. So kàritye, kàriqe") and was dropped from its revised text.
PE22 p. 112 likewise counts q and ty as single consonants for reduplication
only. Medial hy and hw "remained long in PQ ... In TQ the length was reduced"
(PE19 p. 88); they keep weight 2 here, as in Tolkien's definition, and no
attested Third Age word tests the point. Only primary stress is marked,
directly before the stressed vowel, because the text frontend applies it to
the next phone; a long vowel outside the stressed syllable (the ú of únótimë)
would carry secondary stress, which is not marked. Tolkien's rules for
secondary stress (PE19 pp. 58, 60: on alternate syllables outward from the
main stress, attracted by a long syllable not adjacent to it, with a strong
secondary on the first syllable of long compounds) are recorded in the
project's PE19 notes for a front end that keeps them.

Choices made where the sources disagree and Appendix E does not settle the
matter: archaic or Vanyarin spellings (initial w, z, þ, initial ñ) are read
with their Third Age sounds rather than as written (PE19 pp. 71-77 describe
all of these as the Noldor's own pronunciation before the Exile, the archaic
sounds being restorations by the loremasters); plain h between vowels is [h]
(Appendix E: "h in house, behold"; PE19 p. 71: "In the TQ, also medial [χ],
spirantal h, became weakened to breath h"), not the older spirant [x]; ty,
ny, ly are the single palatal phones rather than consonant + [j] (PE19 p. 75:
ty "probably closely similar to t in English tune", explicitly not the [tʃ]
of Vanyarin and Númenórean speech; ny "as in English new"; nothing on ly);
and the digraphs weigh for stress as described above.

Spoken-Tarquesta habits that Tolkien describes in PE19 and that are
deliberately not applied, because the Third Age spelling and Appendix E
follow the careful, learned usage: unstressed long vowels shortened in
speech though spelled long (p. 59); rs and ts said as ss, mn as nn, mb and
nd as mm and nn in colloquial speech (pp. 85, 86, 92, 99); medial sy as
[ʃ] (p. 88); pt as [u̯t] (p. 84); initial ps-, x- as ups-, iks- (p. 79); final
-t of inflexions becoming -s (p. 104). Voiceless stops are "lightly aspirated
(as in English)" (p. 87) and r is "lightly trilled" (pp. 70, 83, 99), which the
phone set cannot express and the voice must supply.

Sources: The Lord of the Rings, Appendix E; Parma Eldalamberon 17 (Words,
Phrases and Passages in The Lord of the Rings, 1950s-60s: Namárië notes
p. 76, Exilic Quenya outline pp. 127-132, Eldarin Roots and Stems p. 154);
Parma Eldalamberon 19 (Quenya Phonology: Outline of Phonetic Development,
1937-38, and Outline of Phonology, c. 1951), 22 and 23; The Road Goes Ever
On; Eldamo (https://eldamo.org/). Page numbers are the journal's printed
pages.

Text handling: words are stressed independently; the article i and the
particles ar, ve, mi are left unstressed. Compounds joined by a hyphen or a
raised dot (Ambar-metta, ohta·caro) are read as separate words, an apostrophe
marks elision (tenn' = tenn), and line breaks, dashes and parentheses become
pauses like a comma.
"""

import re
import unicodedata

SHORT_VOWELS = {'a': 'a', 'e': 'ɛ', 'i': 'i', 'o': 'ɔ', 'u': 'u'}
LONG_VOWELS = {'a': 'aː', 'e': 'eː', 'i': 'iː', 'o': 'oː', 'u': 'uː'}
DIPHTHONGS = {'ai': 'ai', 'au': 'au', 'eu': 'ɛu', 'oi': 'ɔi', 'ui': 'ui', 'iu': 'juː'}

# Small function words that stay unstressed: the article, "and", "like", "in".
UNSTRESSED_WORDS = {'i', 'ar', 've', 'mi'}

STRESS = 'ˈ'

# Accented letter -> (base vowel, long, separate). "separate" marks a dieresis.
_VOWEL_LETTERS = {}
for _base in 'aeiou':
    _VOWEL_LETTERS[_base] = (_base, False, False)
for _base, _acute, _circ, _macron, _dieresis in zip('aeiou', 'áéíóú', 'âêîôû', 'āēīōū', 'äëïöü'):
    _VOWEL_LETTERS[_acute] = (_base, True, False)
    _VOWEL_LETTERS[_circ] = (_base, True, False)
    _VOWEL_LETTERS[_macron] = (_base, True, False)
    _VOWEL_LETTERS[_dieresis] = (_base, False, True)

# Consonant units, longest first: spelling -> (phones, weight). The weight is
# how many consonants the unit counts as for syllable weight.
_CONSONANT_UNITS = [
    ('qu', 'kw', 2),
    ('kw', 'kw', 2),
    ('q', 'kw', 2),
    ('hw', 'ʍ', 2),
    ('hy', 'ç', 2),
    ('hl', 'l', 1),
    ('hr', 'r', 1),
    ('ty', 'c', 2),
    ('ny', 'ɲ', 2),
    ('ly', 'ʎ', 2),
    ('th', 's', 1),
    ('ng', 'ŋɡ', 2),
    ('x', 'ks', 2),
    ('c', 'k', 1),
    ('k', 'k', 1),
    ('g', 'ɡ', 1),
    ('y', 'j', 1),
    ('þ', 's', 1),
    ('ñ', 'ŋ', 1),
    ('z', 'r', 1),
]
_SIMPLE_CONSONANTS = set('bdfhjlmnprstvw')

PUNCTUATION_MARKS = ';:,.!?¡¿…"“”~/。【】、‥،؟؛'
# Marks that the text frontend does not know; they become a pause like a comma.
_PAUSE_MARKS = ('\n', '–', '—', '(', ')', '«', '»')
# Only the letters Quenya spelling uses; anything else separates words.
_WORD_CHARS = "a-zA-ZáéíóúâêîôûäëïöüāēīōūÁÉÍÓÚÂÊÎÔÛÄËÏÖÜĀĒĪŌŪþÞñÑ"
_TOKEN_RE = re.compile(r"[{w}]+(?:['’][{w}]*)*|[{p}]+".format(w=_WORD_CHARS, p=re.escape(PUNCTUATION_MARKS)))


def _tokenize_word(word):
    """
    Splits a lowercase word into segments. A vowel segment is
    ('V', phones, heavy); a consonant segment is ('C', phones, weight).
    """
    segments = []
    i = 0
    while i < len(word):
        char = word[i]
        if char in _VOWEL_LETTERS:
            base, long, _ = _VOWEL_LETTERS[char]
            if not long and i + 1 < len(word) and word[i + 1] in _VOWEL_LETTERS:
                next_base, next_long, next_separate = _VOWEL_LETTERS[word[i + 1]]
                pair = base + next_base
                if pair in DIPHTHONGS and not next_long and not next_separate:
                    segments.append(('V', DIPHTHONGS[pair], True))
                    i += 2
                    continue
            segments.append(('V', LONG_VOWELS[base] if long else SHORT_VOWELS[base], long))
            i += 1
            continue
        for spelling, phones, weight in _CONSONANT_UNITS:
            if word.startswith(spelling, i):
                segments.append(('C', phones, weight))
                i += len(spelling)
                break
        else:
            if char in _SIMPLE_CONSONANTS:
                segments.append(('C', char, 1))
            i += 1  # anything else (digits, stray symbols) is dropped
    return segments


# Diphthongs after which w stays [w] (PE19 p. 72: "after s (z), n, r, l, and
# the diphthongs ai, oi however w remained as such"), as IPA vowel phones.
_W_KEEPING_DIPHTHONGS = {DIPHTHONGS['ai'], DIPHTHONGS['oi']}


def _between_vowels(previous, following):
    """True when a consonant stands between two vowels, the first not ai/oi."""
    return (previous is not None and following is not None
            and previous[0] == 'V' and following[0] == 'V'
            and previous[1] not in _W_KEEPING_DIPHTHONGS)


def _apply_context_rules(segments):
    """
    Rules that depend on neighbouring segments: ht, nasal assimilation,
    initial ñ and ng, w, final ng.
    """
    result = []
    for index, (kind, phones, weight) in enumerate(segments):
        if kind == 'C':
            previous = segments[index - 1] if index > 0 else None
            following = segments[index + 1] if index + 1 < len(segments) else None
            if phones == 'h' and previous is not None and following is not None and following[1][:1] == 'c':
                phones = 'ç'  # hty: the h is fronted whatever the vowel before it
            elif phones == 'h' and previous is not None and following is not None and following[1][:1] == 't':
                phones = 'ç' if previous[0] == 'V' and previous[1].rstrip('ː')[-1] in 'ɛei' else 'x'
            elif phones == 'n' and following is not None and following[1][:1] in ('k', 'ɡ'):
                phones = 'ŋ'
            elif phones == 'ŋɡ' and following is None:
                phones = 'ŋ'
            elif phones in ('ŋ', 'ŋɡ') and previous is None:
                phones = 'n'
            elif phones == 'w' and (previous is None or _between_vowels(previous, following)):
                # Third Age w > v initially and between vowels, kept after
                # consonants and after ai, oi: PE17 pp. 76, 129, 154; PE19
                # p. 72; PE22 p. 44; Appendix E (vilya, older wilya). See
                # the module docstring.
                phones = 'v'
        result.append((kind, phones, weight))
    return result


def _stressed_vowel_index(segments):
    vowel_positions = [i for i, segment in enumerate(segments) if segment[0] == 'V']
    if not vowel_positions:
        return None
    if len(vowel_positions) <= 2:
        return vowel_positions[0]
    penultimate = vowel_positions[-2]
    heavy = segments[penultimate][2]
    if not heavy:
        consonants = sum(weight for _, _, weight in segments[penultimate + 1:vowel_positions[-1]])
        heavy = consonants >= 2
    return penultimate if heavy else vowel_positions[-3]


def quenya_word_to_ipa(word, stressed=True):
    """
    Converts one Quenya word (no spaces or punctuation) to IPA with stress.
    """
    word = unicodedata.normalize('NFC', word).lower()
    word = word.replace("'", "").replace("’", "")
    segments = _apply_context_rules(_tokenize_word(word))
    stress_index = _stressed_vowel_index(segments) if stressed else None
    ipa = ''
    for index, (_, phones, _) in enumerate(segments):
        if index == stress_index:
            # the stress goes on the vowel itself, after the glide of a rising diphthong
            if phones.startswith('j'):
                phones = 'j' + STRESS + phones[1:]
            else:
                phones = STRESS + phones
        ipa += phones
    return ipa


def quenya_to_ipa(word):
    # kept for backwards compatibility with older callers
    return quenya_word_to_ipa(word)


def process_complete_sentence(text):
    """
    Converts a Quenya sentence to IPA. Words are separated by spaces and
    punctuation is attached to the preceding word. Compounds joined by a
    hyphen or a raised dot are read as separate words.
    """
    text = unicodedata.normalize('NFC', text)
    for mark in _PAUSE_MARKS:
        text = text.replace(mark, ', ')
    text = re.sub(r'(?:,\s*){2,}', ', ', text)
    for joiner in ('-', '‐', '·'):
        text = text.replace(joiner, ' ')
    processed_parts = []
    for part in _TOKEN_RE.findall(text.strip()):
        if part[0] in PUNCTUATION_MARKS:
            processed_parts.append(part)
        elif any(char in _VOWEL_LETTERS for char in part.lower()):
            stressed = part.lower().strip("'’") not in UNSTRESSED_WORDS
            processed_parts.append(quenya_word_to_ipa(part, stressed=stressed))
        # a token without a vowel (a stray q, an initial) is not a Quenya word and is skipped

    processed_text = ''
    for i, part in enumerate(processed_parts):
        processed_text += part
        if i + 1 < len(processed_parts) and processed_parts[i + 1][0] not in PUNCTUATION_MARKS:
            processed_text += ' '
    return processed_text.strip(', ')
