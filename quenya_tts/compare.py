"""
Compares Quenya-TTS output with a reference recording of the same text, line by line.

For each line of the text it reports:
- what a phone recogniser hears in the reference and in our output, next to the IPA our rules expect,
  with a phone error rate (PER) and the most common differences;
- the pitch contours of both, aligned in time, with both spectrograms (one PNG per line);
- both versions of the line as separate WAV files, to play back to back.

The phone recogniser is facebook/wav2vec2-xlsr-53-espeak-cv-ft, a multilingual model that writes IPA
instead of words. It has never heard Quenya either, so treat its output as a second opinion, not ground
truth: a difference that shows up in the reference and in our output alike is probably the recogniser.

The reference is cut into lines by running the recogniser over the whole recording and aligning what it
hears with the expected phones of the whole text, so each line is found where its sounds are, even when
the reader runs lines together (--cut pauses cuts at the longest pauses instead). The cut points are
written to segments.csv in the output folder; check them with --segments_only, fix any wrong times, and
run again with --segments.

Needs the reader's requirements plus transformers:
    pip install "transformers>=4.46,<4.50"

Example, from the repository root:
    python -m quenya_tts.compare ~/Music/tolkien_namarie.mp3 --text namarie.txt -o comparisons/namarie

Keep reference recordings you have no rights to (such as Tolkien's) outside the repository.
"""

import argparse
import collections
import csv
import os
import unicodedata

RECOGNIZER = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
SR = 16000  # sample rate for recognition and analysis

# marks that are not phones of their own
_IGNORED = set("ˈˌ.|‖‿ ͜͡-~#,;:!?")
_LENGTH = "ːˑ"
# spelling variants of the same sound in different IPA conventions
_SAME_SOUND = {"g": "ɡ", "ɾ": "r", "ɹ": "r", "ʀ": "r", "ɫ": "l", "ʍ": "hw"}


def ipa_to_phones(ipa, keep_length=False):
    """
    Splits an IPA string into phones. Diacritics and modifier letters stay with the phone before them;
    stress, syllable and pause marks are dropped. Without keep_length, length marks are dropped too.
    """
    phones = []
    for char in unicodedata.normalize("NFD", ipa):
        if char in _IGNORED or char.isspace():
            continue
        if char in _LENGTH:
            if keep_length and phones:
                phones[-1] += "ː"
            continue
        if unicodedata.category(char) == "Mn" or char in "ʰʲʷˠˤⁿˡ":
            if phones:
                phones[-1] += char
            continue
        for part in _SAME_SOUND.get(char, char):
            phones.append(part)
    return [unicodedata.normalize("NFC", phone) for phone in phones]


def align(expected, heard):
    """
    Levenshtein alignment of two phone lists. Returns (distance, pairs), where pairs is a list of
    (expected phone or None, heard phone or None) in order.
    """
    rows, cols = len(expected) + 1, len(heard) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(cost[i - 1][j] + 1,
                             cost[i][j - 1] + 1,
                             cost[i - 1][j - 1] + (expected[i - 1] != heard[j - 1]))
    pairs = []
    i, j = len(expected), len(heard)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (expected[i - 1] != heard[j - 1]):
            pairs.append((expected[i - 1], heard[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((expected[i - 1], None))
            i -= 1
        else:
            pairs.append((None, heard[j - 1]))
            j -= 1
    return cost[-1][-1], pairs[::-1]


def phone_error_rate(expected, heard):
    distance, pairs = align(expected, heard)
    return distance / max(len(expected), 1), [pair for pair in pairs if pair[0] != pair[1]]


def find_lines(intervals, n_lines):
    """
    Groups the non-silent intervals of a recording (sample index pairs, in order) into n_lines lines by
    cutting at the n_lines - 1 longest pauses. Returns a list of (start, end) sample indices.
    """
    if len(intervals) < n_lines:
        raise ValueError(f"found only {len(intervals)} stretches of speech for {n_lines} lines; "
                         f"lower --top_db or give the cut points with --segments")
    gaps = sorted(range(len(intervals) - 1), key=lambda i: intervals[i + 1][0] - intervals[i][1], reverse=True)
    cuts = sorted(gaps[:n_lines - 1])
    lines, start = [], intervals[0][0]
    for i in cuts:
        lines.append((start, intervals[i][1]))
        start = intervals[i + 1][0]
    lines.append((start, intervals[-1][1]))
    return lines


def estimate_lines(intervals, weights):
    """
    Fallback when a recording has fewer pauses than lines: spreads the lines over the speech in proportion
    to their weights (their length in letters), moving each cut to a nearby pause when there is one.
    Returns a list of (start, end) sample indices, one per weight.
    """
    start, end = intervals[0][0], intervals[-1][1]
    pauses = [(intervals[i][1] + intervals[i + 1][0]) // 2 for i in range(len(intervals) - 1)]
    total = float(sum(weights))
    tolerance = (end - start) / len(weights) / 3
    cuts, done = [], 0.0
    for weight in weights[:-1]:
        done += weight
        target = start + (end - start) * done / total
        nearest = min(pauses, key=lambda pause: abs(pause - target), default=None)
        cut = nearest if nearest is not None and abs(nearest - target) <= tolerance else int(target)
        cuts.append(max(cut, cuts[-1] + 1 if cuts else start + 1))
    bounds = [start] + cuts + [end]
    return [(bounds[i], bounds[i + 1]) for i in range(len(weights))]


def cut_reference(reference, lines, top_db):
    """
    Cuts the reference into one span per line. Tries the longest pauses first, with ever lower silence
    thresholds; if the recording still has too few pauses, estimates the cuts. Returns (spans, estimated).
    """
    import librosa

    margin = int(0.1 * SR)
    for threshold in range(int(top_db), 14, -5):
        intervals = librosa.effects.split(reference, top_db=threshold)
        if len(intervals) >= len(lines):
            spans = [(max(0, start - margin), min(len(reference), end + margin))
                     for start, end in find_lines(intervals, len(lines))]
            return spans, False
    intervals = librosa.effects.split(reference, top_db=top_db)
    return estimate_lines(intervals, [len([c for c in line if c.isalpha()]) or 1 for line in lines]), True


def cut_by_phones(line_phones, heard, duration):
    """
    Finds where each line is spoken by aligning the expected phones of the whole text (line_phones: one
    phone list per line) with the phones the recogniser heard in the whole recording (heard: a list of
    (phone, time in seconds)). Each cut goes halfway between the last phone heard for one line and the
    first heard for the next. Returns one (start_s, end_s) per line.
    """
    expected, owner = [], []
    for number, phones in enumerate(line_phones):
        expected += phones
        owner += [number] * len(phones)
    _, pairs = align(expected, [phone for phone, _ in heard])
    first = [None] * len(line_phones)
    last = [None] * len(line_phones)
    e = h = 0
    for exp, got in pairs:
        if exp is not None and got is not None:
            line, time = owner[e], heard[h][1]
            first[line] = time if first[line] is None else first[line]
            last[line] = time
        e += exp is not None
        h += got is not None
    # lines the alignment found nothing for get times interpolated from their neighbours
    known = [i for i in range(len(line_phones)) if first[i] is not None]
    if not known:
        step = duration / len(line_phones)
        return [(i * step, (i + 1) * step) for i in range(len(line_phones))]
    for i in range(len(line_phones)):
        if first[i] is None:
            before = max([k for k in known if k < i], default=None)
            after = min([k for k in known if k > i], default=None)
            low = last[before] if before is not None else 0.0
            high = first[after] if after is not None else duration
            lo_i = before if before is not None else -1
            hi_i = after if after is not None else len(line_phones)
            first[i] = low + (high - low) * (i - lo_i) / (hi_i - lo_i) - 0.01
            last[i] = first[i] + 0.02
    cuts = [(last[i] + first[i + 1]) / 2 for i in range(len(line_phones) - 1)]
    bounds = [max(0.0, first[0] - 0.3)] + cuts + [min(duration, last[-1] + 0.5)]
    return [(bounds[i], max(bounds[i + 1], bounds[i] + 0.05)) for i in range(len(line_phones))]


def snap_to_quiet(cut, loudness, frame_seconds, window=0.4):
    """
    Moves a cut (in seconds) to the quietest frame within window seconds of it. loudness is a sequence of
    per-frame levels (for example RMS) with frame_seconds per frame. Readers usually breathe or pause between
    lines, so this puts a cut that landed on the first sound of the next line back into the gap.
    """
    centre = int(round(cut / frame_seconds))
    reach = int(round(window / frame_seconds))
    low, high = max(0, centre - reach), min(len(loudness), centre + reach + 1)
    if low >= high:
        return cut
    quietest = min(range(low, high), key=lambda i: (loudness[i], abs(i - centre)))
    return quietest * frame_seconds


def find_line_spans(audio, lines, recognize):
    """
    Finds where each line of text is spoken in a recording (audio at SR) with the phone recogniser from
    load_recognizer, moving each cut to the quietest point nearby. Returns one (start_s, end_s) per line.
    """
    import librosa

    from quenya_g2p import to_ipa

    line_phones = [ipa_to_phones(to_ipa(line)) for line in lines]
    found = cut_by_phones(line_phones, recognize.timed(audio), len(audio) / SR)
    hop = 160
    loudness = librosa.feature.rms(y=audio, frame_length=640, hop_length=hop)[0].tolist()
    cuts = [snap_to_quiet(end, loudness, hop / SR) for _, end in found[:-1]]
    bounds = [found[0][0]] + cuts + [found[-1][1]]
    return [(bounds[i], max(bounds[i + 1], bounds[i] + 0.05)) for i in range(len(lines))]


def load_recognizer(device):
    import torch
    from transformers import AutoModelForCTC
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(RECOGNIZER)
    model = AutoModelForCTC.from_pretrained(RECOGNIZER).to(device).eval()

    def recognize(audio):
        inputs = processor(audio, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        return processor.batch_decode(torch.argmax(logits, dim=-1))[0]

    def recognize_timed(audio):
        """The phones heard in a long recording, each with the time in seconds where it starts."""
        seconds_per_frame = model.config.inputs_to_logits_ratio / SR
        heard, previous = [], None
        chunk = 30 * SR  # recognise in chunks, to keep memory low on long recordings
        for offset in range(0, len(audio), chunk):
            inputs = processor(audio[offset:offset + chunk], sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                ids = torch.argmax(model(inputs.input_values.to(device)).logits, dim=-1)[0].tolist()
            for frame, token_id in enumerate(ids):
                if token_id != previous and token_id != processor.tokenizer.pad_token_id:
                    time = offset / SR + frame * seconds_per_frame
                    token = processor.tokenizer.convert_ids_to_tokens(token_id)
                    heard += [(phone, time) for phone in ipa_to_phones(token)]
                previous = token_id
        return heard

    recognize.timed = recognize_timed
    return recognize


def plot_line(reference, ours, title, path):
    """Spectrograms of both versions and their pitch contours, with ours warped onto the reference's timing."""
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    hop = 160
    mfcc_ref = librosa.feature.mfcc(y=reference, sr=SR, n_mfcc=20, hop_length=hop)
    mfcc_ours = librosa.feature.mfcc(y=ours, sr=SR, n_mfcc=20, hop_length=hop)
    _, path_pairs = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_ours, metric="cosine")

    def pitch(audio):
        f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=500, sr=SR, hop_length=hop)
        f0 = np.where(voiced, f0, np.nan)
        median = np.nanmedian(f0) if np.any(voiced) else 1.0
        return 12 * np.log2(f0 / median)  # semitones around the speaker's own median

    pitch_ref, pitch_ours = pitch(reference), pitch(ours)
    warped = np.full(len(pitch_ref), np.nan)
    for i, j in path_pairs[::-1]:
        if i < len(warped) and j < len(pitch_ours):
            warped[i] = pitch_ours[j]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    for ax, audio, label in ((axes[0], reference, "reference"), (axes[1], ours, "Quenya-TTS")):
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=512, hop_length=hop)), ref=np.max)
        librosa.display.specshow(spec, sr=SR, hop_length=hop, x_axis="time", y_axis="hz", ax=ax)
        ax.set_ylim(0, 5000)
        ax.set_title(label)
    times = librosa.frames_to_time(np.arange(len(pitch_ref)), sr=SR, hop_length=hop)
    axes[2].plot(times, pitch_ref, label="reference")
    axes[2].plot(times, warped, label="Quenya-TTS, aligned to the reference")
    axes[2].set_ylabel("semitones from median")
    axes[2].set_xlabel("time in the reference (s)")
    axes[2].legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare Quenya-TTS output with a reference recording.")
    parser.add_argument("reference", help="the reference recording (wav, mp3, flac, ...)")
    parser.add_argument("--text", required=True, help="the text, one line per line, in the order it is spoken")
    parser.add_argument("-o", "--output", default="comparisons/latest", help="folder for the report, plots and WAVs")
    parser.add_argument("--segments", help="CSV with start_s,end_s per line (as written to segments.csv) instead of cutting at pauses")
    parser.add_argument("--cut", choices=["phones", "pauses"], default="phones",
                        help="find the lines where the recogniser hears their phones (default), or at the longest pauses")
    parser.add_argument("--top_db", type=float, default=35, help="how far below the peak counts as silence when cutting lines")
    parser.add_argument("--model", default=None, help="ToucanTTS checkpoint (.pt); default: the pretrained multilingual model")
    parser.add_argument("--speaker", default=None, help="a WAV file whose voice our model should imitate (try the reference)")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--speed", type=float, default=1.0, help="our duration scaling, as in quenya_tts.read")
    parser.add_argument("--pitch_variance", type=float, default=1.0, help="our melody range, as in quenya_tts.read")
    parser.add_argument("--creativity", type=float, default=0.0, help="our prosody temperature, as in quenya_tts.read")
    parser.add_argument("--segments_only", action="store_true",
                        help="only cut the reference into lines (segments.csv and one WAV per line), to check the cuts")
    parser.add_argument("--no_plots", action="store_true", help="skip the pitch plots, which are the slow part")
    args = parser.parse_args()

    import librosa
    import numpy as np
    import soundfile

    from quenya_g2p import to_ipa
    from quenya_tts.read import load_reader
    from quenya_tts.read import synthesize
    from quenya_tts.watermark import write_generated

    with open(args.text, encoding="utf8") as f:
        lines = [line.strip() for line in f if line.strip()]
    os.makedirs(args.output, exist_ok=True)
    speaker = os.path.abspath(args.speaker) if args.speaker else None
    model = os.path.abspath(args.model) if args.model else None

    reference, _ = librosa.load(args.reference, sr=SR, mono=True)
    print("Loading the phone recogniser")
    recognize = load_recognizer(args.device)
    if args.segments:
        with open(args.segments, encoding="utf8") as f:
            rows = list(csv.DictReader(f))
        spans = [(int(float(row["start_s"]) * SR), int(float(row["end_s"]) * SR)) for row in rows]
        if len(spans) != len(lines):
            parser.error(f"{args.segments} has {len(spans)} rows for {len(lines)} lines of text")
    elif args.cut == "pauses":
        spans, estimated = cut_reference(reference, lines, args.top_db)
        if estimated:
            print("The recording has fewer pauses than the text has lines, so some cut points are estimated. "
                  "Check segments.csv and fix any wrong times, then run again with --segments.")
    else:
        print("Finding the lines in the reference")
        spans = [(int(start * SR), int(end * SR)) for start, end in find_line_spans(reference, lines, recognize)]
    with open(os.path.join(args.output, "segments.csv"), "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["line", "start_s", "end_s", "text"])
        for number, ((start, end), line) in enumerate(zip(spans, lines), 1):
            writer.writerow([number, f"{start / SR:.2f}", f"{end / SR:.2f}", line])
    if args.segments_only:
        print(f"Cut points: {os.path.join(args.output, 'segments.csv')}")
        for number, (start, end) in enumerate(spans, 1):
            soundfile.write(os.path.join(args.output, f"line{number:02d}_reference.wav"), reference[start:end], SR)
        return

    print("Loading the speech synthesiser")
    tts, phonemize = load_reader(model=model, speaker=speaker, device=args.device)

    results = []
    differences = {"reference": collections.Counter(), "ours": collections.Counter()}
    for number, ((start, end), line) in enumerate(zip(spans, lines), 1):
        print(f"Line {number}: {line}")
        ref_audio = reference[start:end]
        wave, sr = synthesize(tts, phonemize(line), speed=args.speed,
                              pitch_variance=args.pitch_variance, creativity=args.creativity)
        ours = librosa.resample(np.asarray(wave, dtype=np.float32), orig_sr=sr, target_sr=SR)
        ours = librosa.effects.trim(ours, top_db=args.top_db)[0]

        soundfile.write(os.path.join(args.output, f"line{number:02d}_reference.wav"), ref_audio, SR)
        write_generated(os.path.join(args.output, f"line{number:02d}_quenya_tts.wav"), ours, SR)

        expected_ipa = to_ipa(line)
        expected = ipa_to_phones(expected_ipa)
        row = {"number": number, "text": line, "expected": expected_ipa,
               "ratio": (len(ours) / SR) / max((end - start) / SR, 0.01)}
        for key, audio in (("reference", ref_audio), ("ours", ours)):
            heard = recognize(audio)
            per, diffs = phone_error_rate(expected, ipa_to_phones(heard))
            differences[key].update(diffs)
            row[key] = heard
            row[key + "_per"] = per
        results.append(row)

        if not args.no_plots:
            plot_line(ref_audio, ours, f"{number}. {line}", os.path.join(args.output, f"line{number:02d}.png"))

    write_report(args, results, differences)
    print(f"Report: {os.path.join(args.output, 'report.md')}")


def _describe(pair):
    expected, heard = pair
    if expected is None:
        return f"extra {heard}"
    if heard is None:
        return f"missing {expected}"
    return f"{expected} heard as {heard}"


def write_report(args, results, differences):
    def mean(key):
        return sum(row[key] for row in results) / max(len(results), 1)

    out = [f"# Comparison with {os.path.basename(args.reference)}", "",
           f"Phone recogniser: {RECOGNIZER}. PER is the phone error rate against the IPA from our rules "
           "(stress and length marks ignored). Length ratio is our duration divided by the reference's.", "",
           "| | Reference | Quenya-TTS |", "|---|---|---|",
           f"| Mean PER | {mean('reference_per'):.0%} | {mean('ours_per'):.0%} |",
           f"| Mean length ratio | | {mean('ratio'):.2f} |", ""]
    for key, label in (("reference", "reference"), ("ours", "Quenya-TTS output")):
        out += [f"## Most common differences in the {label}", ""]
        out += [f"- {_describe(pair)}: {count}" for pair, count in differences[key].most_common(15)] or ["- none"]
        out.append("")
    out += ["## Lines", ""]
    for row in results:
        n = row["number"]
        out += [f"### {n}. {row['text']}", "",
                f"- Our rules: `{row['expected']}`",
                f"- Heard in the reference: `{row['reference']}` (PER {row['reference_per']:.0%})",
                f"- Heard in Quenya-TTS: `{row['ours']}` (PER {row['ours_per']:.0%})",
                f"- Length ratio: {row['ratio']:.2f}",
                f"- Audio: [reference](line{n:02d}_reference.wav), [Quenya-TTS](line{n:02d}_quenya_tts.wav)", ""]
        if not args.no_plots:
            out += [f"![pitch and spectrograms](line{n:02d}.png)", ""]
    with open(os.path.join(args.output, "report.md"), "w", encoding="utf8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
