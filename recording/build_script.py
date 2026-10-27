"""
Builds the recording script (recording/script.tsv) for a volunteer voice.

The script mixes four kinds of lines, all in Third Age spelling:
- book: the recorded lines of the Ma Quetilyë Quenya? book (audio/parande_N/SCRIPT.md), recorded first
  because the book needs them anyway;
- phrase: short attested Quenya phrases from Eldamo;
- words: attested Quenya words from Eldamo, four to a line; verbs as the 1st person aorist (tirin, valtan);
- sentence: short sentences built from attested words with regular grammar ("I cirya ná ninquë.").
  The grammar is simple on purpose; the words are all attested, the sentences are not.

After the book lines, every line is picked for the sounds it adds: lines with phones, phone pairs and
stress patterns the script does not cover well yet come first. So the first sessions already cover every
sound, and a speaker who stops early still leaves a useful corpus.

Lines are numbered in sessions of 100 (s01_001, s01_002, ...). Record each line as its own file named by
its id, see recording/README.md.

Eldamo (https://eldamo.org) is © Paul Strack, CC BY 4.0. Get the data file from
https://raw.githubusercontent.com/pfstrack/eldamo/master/src/data/eldamo-data.xml

Example, from the repository root:
    python recording/build_script.py --eldamo eldamo-data.xml --book ../ma-quetelye-quenya/audio
"""

import argparse
import collections
import glob
import os
import random
import re
import unicodedata
import xml.etree.ElementTree as ET

from quenya_g2p import to_ipa
from quenya_tts.compare import ipa_to_phones

LINES_PER_SESSION = 100
WORDS_PER_LINE = 4
# Eldamo word classes read as words; names are included for their sounds
WORD_CLASSES = {"n", "adj", "adv", "vb", "interj", "pron", "cardinal", "ordinal", "conj", "prep",
                "masc-name", "fem-name", "place-name", "proper-name", "collective-name"}
# Letters of archaic or alternative spellings: the book writes Third Age Quenya with c, s, r and qu
ARCHAIC = set("kþθχzöäüǭ")
MAX_PHRASE_WORDS = 7
TEMPLATES = [
    "I {noun} ná {adj}.",
    "Ma i {noun} ná {adj}?",
    "Cenin i {noun}.",
    "I {noun} ar i {noun2}.",
]


def clean(form):
    """Eldamo form -> spelling for reading: no homonym numbers, optional letters dropped, NFC."""
    form = re.sub(r"[¹²³⁴]", "", form)
    form = re.sub(r"\([^)]*\)", "", form)
    return unicodedata.normalize("NFC", form.strip())


def readable(text):
    lowered = text.lower()
    return (text and not ARCHAIC & set(lowered) and not re.search(r"[\[\]{}<>‽#*|/=+]|\?.|(^|\s)-|-(\s|$)", text)
            and re.fullmatch(r"[a-zà-ÿ' ,.!?—-]+", lowered))


def verb_form(stem):
    """1st person aorist: a-stems add -n (valta- > valtan), basic stems -in (tir- > tirin)."""
    stem = stem.rstrip("-")
    return stem + "n" if stem.endswith("a") else stem + "in"


def attested(word):
    return word.get("l") == "q" and not word.get("mark")


def eldamo_entries(path):
    root = ET.parse(path).getroot()
    words, phrases, nouns, adjectives = [], [], [], []
    for word in root.iter("word"):
        if not attested(word):
            continue
        form, speech, gloss = clean(word.get("v", "")), word.get("speech"), word.get("gloss", "")
        if not readable(form.rstrip("-")):
            continue
        if speech == "phrase" and 2 <= len(form.split()) <= MAX_PHRASE_WORDS:
            phrases.append((form[0].upper() + form[1:].rstrip(",") + ("" if form[-1] in ".!?" else "."),
                            f"eldamo:{word.get('page-id')}", gloss))
        elif speech in WORD_CLASSES and " " not in form and not form.startswith("-"):
            if speech == "vb":
                if not form.endswith("-"):
                    continue
                form = verb_form(form)
            elif form.endswith("-"):
                continue
            words.append((form, word.get("page-id"), gloss))
            if speech == "n" and form.islower():
                nouns.append((form, gloss))
            if speech == "adj" and form[-1] in "aëe":
                adjectives.append((form, gloss))
    return dedupe(words), dedupe(phrases), dedupe(nouns), dedupe(adjectives)


def dedupe(items):
    seen, kept = set(), []
    for item in items:
        if item[0].lower() not in seen:
            seen.add(item[0].lower())
            kept.append(item)
    return kept


def book_lines(book_audio_dir):
    """Lines of every audio/parande_N/SCRIPT.md: one dialogue turn or list item per line."""
    lines = []
    for path in sorted(glob.glob(os.path.join(book_audio_dir, "parande_*", "SCRIPT.md"))):
        chapter = os.path.basename(os.path.dirname(path)).split("_")[1]
        track = None
        with open(path, encoding="utf8") as f:
            for raw in f:
                heading = re.match(r"## (\d+\.\d+)", raw)
                if heading:
                    track = heading.group(1)
                if not raw.startswith(">") or track is None:
                    continue
                text = re.sub(r"\*\*[^*]+:\*\*", "", raw[1:]).strip()
                for part in re.split(r"\s+…\s+|\s+·\s+", text):
                    part = re.sub(r"^\w{1,2}:\s+", "", part.strip())  # "á: mára, málo" in the sounds table
                    part = part.replace("*", "").strip()
                    if part and readable(part):
                        lines.append((part, f"book:{chapter}/{track}", ""))
    return dedupe(lines)


def word_lines(words, rng):
    words = list(words)
    rng.shuffle(words)
    lines = []
    for i in range(0, len(words) - WORDS_PER_LINE + 1, WORDS_PER_LINE):
        group = words[i:i + WORDS_PER_LINE]
        text = ", ".join(w for w, _, _ in group)
        lines.append((text[0].upper() + text[1:] + ".", "eldamo:" + " ".join(p for _, p, _ in group),
                      "; ".join(g for _, _, g in group)))
    return lines


def sentence_lines(nouns, adjectives, count, rng):
    lines, seen = [], set()
    while len(lines) < count:
        template = rng.choice(TEMPLATES)
        (noun, noun_gloss), (noun2, noun2_gloss) = rng.sample(nouns, 2)
        adj, adj_gloss = rng.choice(adjectives)
        text = template.format(noun=noun, noun2=noun2, adj=adj)
        if text in seen:
            continue
        seen.add(text)
        gloss = {"I {noun} ná {adj}.": f"the {noun_gloss} is {adj_gloss}",
                 "Ma i {noun} ná {adj}?": f"is the {noun_gloss} {adj_gloss}?",
                 "Cenin i {noun}.": f"I see the {noun_gloss}",
                 "I {noun} ar i {noun2}.": f"the {noun_gloss} and the {noun2_gloss}"}[template]
        lines.append((text, "sentence", gloss))
    return lines


DIPHTHONGS = re.compile(r"ai|au|ɔi|ui|ɛu|iu")


def syllables(ipa_word):
    return len(re.findall(r"[aeiouɛɔ]", DIPHTHONGS.sub("a", ipa_word)))


def stress_patterns(ipa):
    """One feature per word: its syllable count (up to 5) and which syllable from the end is stressed."""
    patterns = []
    for word in re.sub(r"[,.!?;:]", " ", ipa).split():
        if "ˈ" in word:
            patterns.append(f"stress:{min(syllables(word), 5)}/{syllables(word.split('ˈ', 1)[1])}")
    return patterns


def features(text):
    ipa = to_ipa(text)
    phones = ipa_to_phones(ipa, keep_length=True)
    pairs = [f"{a}+{b}" for a, b in zip(phones, phones[1:])]
    return phones, pairs + stress_patterns(ipa)


def order_by_coverage(candidates, covered, total):
    """Greedy: pick the line that adds the most still-rare features per phone, until `total` lines."""
    scored = [(line, *features(line[0])) for line in candidates]
    chosen = []
    while scored and len(chosen) < total:
        best, best_gain = None, -1.0
        for i, (line, phones, extra) in enumerate(scored):
            items = phones + extra
            gain = sum(1.0 / (1 + covered[item]) for item in items) / max(len(phones), 1) ** 0.5
            if gain > best_gain:
                best, best_gain = i, gain
        line, phones, extra = scored.pop(best)
        covered.update(phones + extra)
        chosen.append(line)
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Build the recording script for a volunteer voice.")
    parser.add_argument("--eldamo", required=True, help="eldamo-data.xml")
    parser.add_argument("--book", required=True, help="the book repository's audio/ folder")
    parser.add_argument("--lines", type=int, default=2500, help="number of lines (about 2.5 seconds each, with pauses)")
    parser.add_argument("--sentences", type=int, default=1500, help="template sentences among the candidates")
    parser.add_argument("-o", "--output", default=os.path.join(os.path.dirname(__file__), "script.tsv"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    words, phrases, nouns, adjectives = eldamo_entries(args.eldamo)
    book = book_lines(args.book)
    candidates = phrases + word_lines(words, rng) + sentence_lines(nouns, adjectives, args.sentences, rng)
    print(f"book {len(book)}, phrases {len(phrases)}, words {len(words)}, nouns {len(nouns)}, "
          f"adjectives {len(adjectives)}, candidates {len(candidates)}")

    covered = collections.Counter()
    for text, _, _ in book:
        phones, extra = features(text)
        covered.update(phones + extra)
    lines = book + order_by_coverage(candidates, covered, args.lines - len(book))

    with open(args.output, "w", encoding="utf8") as f:
        f.write("id\ttext\tsource\tgloss\n")
        for n, (text, source, gloss) in enumerate(lines):
            session, number = divmod(n, LINES_PER_SESSION)
            f.write(f"s{session + 1:02d}_{number + 1:03d}\t{text}\t{source}\t{gloss}\n")
    kinds = collections.Counter(source.split(":")[0] if ":" in source else source for _, source, _ in lines)
    phone_counts = collections.Counter(p for text, _, _ in lines for p in features(text)[0])
    print(f"{len(lines)} lines in {args.output}: {dict(kinds)}")
    print("rarest phones:", phone_counts.most_common()[-12:])


if __name__ == "__main__":
    main()
