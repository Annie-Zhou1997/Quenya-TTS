"""
Builds a training corpus from recordings of whole texts (poems, prayers, short passages) and their Quenya text.

Each recording is cut into one clip per line of its text, the same way quenya_tts.compare finds lines in a
reference: a phone recogniser listens to the whole recording, and its phones are aligned with the phones our
rules expect for the whole text. The clips and a transcript.txt ("file_name|Quenya text" per line) are
written to Corpora/Quenya/<name> (or $QUENYA_CORPORA_DIR/<name>), ready for quenya_tts.train.

Recordings and texts are paired by file name: namarie.mp3 goes with namarie.txt, which has one line of
Quenya per line, in the order it is read. A line whose clip the recogniser hears too differently from the
expected phones (--max_per) is probably cut wrongly or read differently, so it is left out; cuts.csv in the
corpus folder lists every line with its times, its phone error rate and whether it was kept.

Recordings made from the recording script (recording/script.tsv) are already one file per line, named by the
line's id (s01_001.wav, ...). Pass --script instead of --texts for those: nothing is cut, every file is
checked the same way and stored as <speaker>_<id>.wav.

Needs the reader's requirements plus transformers, like quenya_tts.compare:
    pip install "transformers>=4.46,<4.50"

Example, from the repository root:
    python -m quenya_tts.prepare_corpus --audio ~/glaemscrafu --texts ~/glaemscrafu/texts --name Glaemscrafu
    python -m quenya_tts.prepare_corpus --audio ~/recordings --script recording/script.tsv --speaker bruno --name Volunteers
"""

import argparse
import csv
import os

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")


def pair_files(audio_dir, text_dir):
    """Returns sorted (stem, audio path, text path) for every recording that has a text with the same stem."""
    texts = {os.path.splitext(name)[0]: os.path.join(text_dir, name)
             for name in os.listdir(text_dir) if name.endswith(".txt")}
    pairs = []
    for name in sorted(os.listdir(audio_dir)):
        stem, extension = os.path.splitext(name)
        if extension.lower() in AUDIO_EXTENSIONS and stem in texts:
            pairs.append((stem, os.path.join(audio_dir, name), texts[stem]))
    return pairs


def read_script(path):
    """Returns {line id: text} from a recording script (tab-separated, with id and text columns)."""
    with open(path, encoding="utf8", newline="") as f:
        return {row["id"]: row["text"] for row in csv.DictReader(f, delimiter="\t")}


def pair_script(audio_dir, script):
    """Returns sorted (id, audio path, text) for every recording named after a line of the script."""
    pairs = []
    for name in sorted(os.listdir(audio_dir)):
        stem, extension = os.path.splitext(name)
        if extension.lower() in AUDIO_EXTENSIONS and stem in script:
            pairs.append((stem, os.path.join(audio_dir, name), script[stem]))
    return pairs


def read_lines(path):
    with open(path, encoding="utf8") as f:
        return [line.strip() for line in f if line.strip()]


def clip_name(stem, number):
    return f"{stem}_{number:02d}.wav"


def write_transcript(path, rows):
    """Writes transcript.txt from (file name, text) rows."""
    with open(path, "w", encoding="utf8") as f:
        for file_name, text in rows:
            f.write(f"{file_name}|{text}\n")


def main():
    parser = argparse.ArgumentParser(description="Cut recordings of Quenya texts into a training corpus.")
    parser.add_argument("--audio", required=True, help="folder with the recordings (mp3, wav, flac, ...)")
    parser.add_argument("--texts", help="folder with one .txt per recording, one line of Quenya per line")
    parser.add_argument("--script", help="recording script (recording/script.tsv), for one file per line named by its id")
    parser.add_argument("--speaker", help="with --script: the speaker's name, put in front of every file name")
    parser.add_argument("--name", required=True, help="corpus name, as quenya_tts.train expects it (e.g. Glaemscrafu)")
    parser.add_argument("--max_per", type=float, default=0.6,
                        help="leave out lines whose phone error rate is above this (probably cut or read wrongly)")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    args = parser.parse_args()
    if bool(args.texts) == bool(args.script):
        parser.error("pass either --texts (whole texts to cut into lines) or --script (one file per line)")
    if args.script and not args.speaker:
        parser.error("--script needs --speaker")

    import librosa
    import soundfile

    from quenya_g2p import to_ipa
    from quenya_tts.compare import SR, find_line_spans, ipa_to_phones, load_recognizer, phone_error_rate
    from quenya_tts.corpora import corpus_root

    if args.script:
        pairs = [(f"{args.speaker}_{line_id}", path, [text])
                 for line_id, path, text in pair_script(args.audio, read_script(args.script))]
    else:
        pairs = [(stem, path, read_lines(text_path)) for stem, path, text_path in pair_files(args.audio, args.texts)]
    if not pairs:
        parser.error(f"no recording in {args.audio} matches " + (args.script or f"a .txt in {args.texts}"))
    root = corpus_root(args.name)
    os.makedirs(root, exist_ok=True)

    print("Loading the phone recogniser")
    recognize = load_recognizer(args.device)

    kept, report = [], []
    for stem, audio_path, lines in pairs:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        spans = find_line_spans(audio_16k, lines, recognize) if len(lines) > 1 else [(0.0, len(audio) / sr)]
        kept_here = 0
        for number, ((start, end), line) in enumerate(zip(spans, lines), 1):
            clip = audio[int(start * sr):int(end * sr)]
            heard = recognize(audio_16k[int(start * SR):int(end * SR)])
            per, _ = phone_error_rate(ipa_to_phones(to_ipa(line)), ipa_to_phones(heard))
            keep = per <= args.max_per
            name = clip_name(stem, number) if len(lines) > 1 else f"{stem}.wav"
            if keep:
                soundfile.write(os.path.join(root, name), clip, sr)
                kept.append((name, line))
                kept_here += 1
            report.append([name, f"{start:.2f}", f"{end:.2f}", f"{per:.2f}", "yes" if keep else "no", line, heard])
        if len(lines) > 1 or not kept_here:
            print(f"{stem}: {kept_here} of {len(lines)} lines kept")

    # keep what other runs (other speakers or recordings) already put in this corpus
    transcript_path = os.path.join(root, "transcript.txt")
    done = {row[0] for row in report}
    earlier = []
    if os.path.exists(transcript_path):
        with open(transcript_path, encoding="utf8") as f:
            earlier = [line.rstrip("\n").split("|", 1) for line in f if "|" in line]
    write_transcript(transcript_path, [row for row in earlier if row[0] not in done] + kept)
    with open(os.path.join(root, "cuts.csv"), "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "start_s", "end_s", "per", "kept", "text", "heard"])
        writer.writerows(report)
    seconds = sum(float(row[2]) - float(row[1]) for row in report if row[4] == "yes")
    print(f"{len(kept)} clips, {seconds / 60:.1f} minutes, in {root} (details in cuts.csv)")


if __name__ == "__main__":
    main()
