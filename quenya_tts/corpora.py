"""
Quenya corpora as {audio path: Toucan phoneme string} dicts, ready for Toucan's phoneme-input training.

Each corpus is a folder with transcript.txt (one "file_name|Quenya text" per line) and the audio files next
to it, under Corpora/Quenya/<name> or $QUENYA_CORPORA_DIR/<name>.
"""

import os

from quenya_tts.toucan import CORPORA_DIR

# Corpora whose terms allow publishing a model trained on them: the volunteer recordings, given for training
PUBLIC_CORPORA = ["Volunteers"]
# The thesis recordings: Atanquesta is CC BY-NC-SA 4.0, so models trained on these are for experiments only
EXPERIMENTAL_CORPORA = ["Atanquesta", "Glaemscrafu"]


def corpus_root(name):
    return os.path.join(os.environ.get("QUENYA_CORPORA_DIR", os.path.join(CORPORA_DIR, "Quenya")), name)


def read_transcripts(name):
    """Returns {absolute audio path: Quenya text} for one corpus."""
    root = corpus_root(name)
    path_to_text = dict()
    with open(os.path.join(root, "transcript.txt"), encoding="utf8") as f:
        for line in f:
            if line.strip() and "|" in line:
                file_name, text = line.split("|", 1)
                path_to_text[os.path.abspath(os.path.join(root, file_name.strip()))] = text.strip()
    return path_to_text


def read_phonemized(name, phonemize):
    """Returns {absolute audio path: phoneme string} for one corpus, using a quenya_tts.toucan.QuenyaPhonemizer."""
    return {path: phonemize(text) for path, text in read_transcripts(name).items()}
