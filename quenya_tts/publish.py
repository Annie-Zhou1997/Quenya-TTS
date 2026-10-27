"""
Prepares a trained Quenya model for release, so that it speaks with an artificial voice and not with the voice
of any volunteer.

Toucan saves the speaker of the first training clip as a checkpoint's default voice. A checkpoint released as it
is would therefore read with a volunteer's voice. This step replaces that voice with an artificial one from
Toucan's voice generator (a GAN that makes speaker embeddings for voices that do not exist), after checking that
it does not sound like any volunteer: its speaker embedding must be less similar to each volunteer's than
SpeechBrain's own same-speaker threshold. The released file holds only the model weights, its config and that
voice, next to the model card, the licence and the credits (the names volunteers gave on the site).

It also refuses to release a model trained on anything but the public corpora (training.json, written by
quenya_tts.train).

    python -m quenya_tts.volunteers download -o ~/volunteers            # the voices to compare with
    python -m quenya_tts.publish Models/ToucanTTS_Quenya/best.pt --voices ~/volunteers -o release/

Then check a reading of the release, which must say "watermarked by Quenya-TTS":
    python -m quenya_tts.read "Elen síla lúmenn' omentielvo" --model release/quenya_tts.pt -o release/sample.wav
    python -m quenya_tts.watermark check release/sample.wav
"""

import argparse
import datetime
import glob
import json
import os
import shutil

import numpy as np

# SpeechBrain's verification threshold for its ECAPA speaker embeddings: above this cosine similarity two
# recordings count as the same speaker. The released voice must stay below it for every volunteer.
SAME_SPEAKER_THRESHOLD = 0.25
# How many artificial voices to try, and how many recordings per volunteer to compare with
CANDIDATES = 50
CLIPS_PER_VOLUNTEER = 20
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_FILES = ["MODEL_CARD.md", "MODEL_LICENSE.md"]


def training_info(checkpoint):
    """The training.json next to a checkpoint, or None."""
    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "training.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf8") as f:
        return json.load(f)


def release_problem(info):
    """Why a model with this training.json cannot be released, or None."""
    from quenya_tts.corpora import PUBLIC_CORPORA
    if info is None:
        return ("there is no training.json next to the checkpoint, so it is unknown what it was trained on; "
                "train it with quenya_tts.train")
    other = sorted(set(info.get("corpora", [])) - set(PUBLIC_CORPORA))
    if other or not info.get("public"):
        return f"it was trained on corpora that may not be published: {', '.join(other) or 'unknown'}"
    return None


def volunteer_clips(folder, per_volunteer=CLIPS_PER_VOLUNTEER):
    """{volunteer: [wav paths]} from a quenya_tts.volunteers download folder (<volunteer>/<recording>/<line>.wav)."""
    clips = dict()
    for volunteer in sorted(os.listdir(folder)):
        paths = sorted(glob.glob(os.path.join(folder, volunteer, "*", "*.wav")))
        if paths:
            # spread over all their recordings, not only the first
            step = max(1, len(paths) // per_volunteer)
            clips[volunteer] = paths[::step][:per_volunteer]
    return clips


def credit_names(folder):
    """The names volunteers gave to be credited by, from the download's volunteers.tsv, sorted and without repeats."""
    import csv
    path = os.path.join(folder, "volunteers.tsv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf8", newline="") as f:
        return sorted({row["credit_name"].strip() for row in csv.DictReader(f, delimiter="\t") if row["credit_name"].strip()})


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def pick_voice(candidates, volunteers, threshold=SAME_SPEAKER_THRESHOLD):
    """
    Returns (index, highest similarity) of the first candidate embedding whose cosine similarity to every
    volunteer embedding is below threshold. Raises ValueError if none is.
    """
    best = None
    for index, candidate in enumerate(candidates):
        highest = max((cosine(candidate, voice) for voice in volunteers), default=-1.0)
        if highest < threshold:
            return index, highest
        best = highest if best is None else min(best, highest)
    raise ValueError(f"none of the {len(candidates)} artificial voices is far enough from every volunteer "
                     f"(closest: {best:.2f}, needed below {threshold})")


def speaker_embedding(paths, encoder, device="cpu"):
    """The mean ECAPA speaker embedding of some recordings, as Toucan computes it for a reference voice."""
    import librosa
    import soundfile
    import torch
    from torchaudio.transforms import Resample
    embeddings = []
    for path in paths:
        wave, sr = soundfile.read(path, dtype="float32", always_2d=True)
        wave = Resample(orig_freq=sr, new_freq=16000)(torch.tensor(librosa.to_mono(wave.T)))
        embeddings.append(encoder.encode_batch(wavs=wave.to(device).unsqueeze(0)).squeeze().cpu())
    return sum(embeddings) / len(embeddings)


def artificial_voices(count, device="cpu"):
    """Speaker embeddings of count artificial voices from Toucan's embedding GAN (the same for every run)."""
    import torch
    from huggingface_hub import hf_hub_download
    from Modules.ControllabilityGAN.GAN import GanWrapper
    from Utility.storage_config import MODEL_DIR
    path = hf_hub_download(cache_dir=MODEL_DIR, repo_id="Flux9665/ToucanTTS", filename="embedding_gan.pt")
    gan = GanWrapper(path, device=device, num_cached_voices=count)
    voices = []
    for seed in range(count):
        gan.set_latent(seed)
        # the controllability sliders all at 0: the voice as the GAN made it
        voices.append(gan.modify_embed(torch.zeros(6)).squeeze().cpu())
    return voices


def main():
    parser = argparse.ArgumentParser(description="Prepare a trained Quenya model for release with an artificial voice.")
    parser.add_argument("checkpoint", help="the trained model, e.g. Models/ToucanTTS_Quenya/best.pt")
    parser.add_argument("--voices", required=True,
                        help="a quenya_tts.volunteers download folder: the voices the released voice must not resemble")
    parser.add_argument("-o", "--output", default="release")
    parser.add_argument("--threshold", type=float, default=SAME_SPEAKER_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    problem = release_problem(training_info(args.checkpoint))
    if problem:
        parser.exit(1, f"Not releasing {args.checkpoint}: {problem}.\n")
    clips = volunteer_clips(args.voices)
    if not clips:
        parser.exit(1, f"No volunteer recordings in {args.voices}: download them first (quenya_tts.volunteers).\n")
    checkpoint_path = os.path.abspath(args.checkpoint)
    output = os.path.abspath(args.output)

    from quenya_tts.toucan import setup_toucan
    setup_toucan()
    import torch
    from speechbrain.pretrained import EncoderClassifier
    from Utility.storage_config import MODEL_DIR

    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": args.device},
                                             savedir=os.path.join(MODEL_DIR, "Embedding", "speechbrain_speaker_embedding_ecapa"))
    volunteers = [speaker_embedding(paths, encoder, args.device) for paths in clips.values()]
    candidates = artificial_voices(CANDIDATES, args.device)
    try:
        index, highest = pick_voice(candidates, volunteers, args.threshold)
    except ValueError as err:
        parser.exit(1, f"Not releasing: {err}.\n")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    os.makedirs(output, exist_ok=True)
    torch.save({"model": checkpoint["model"], "config": checkpoint["config"], "default_emb": candidates[index]},
               os.path.join(output, "quenya_tts.pt"))
    for name in RELEASE_FILES:
        shutil.copy(os.path.join(REPO_DIR, name), output)
    with open(os.path.join(output, "CREDITS.txt"), "w", encoding="utf8") as f:
        f.write("Voices: the volunteers of Quenya Voices, among them\n")
        f.writelines(f"- {name}\n" for name in credit_names(args.voices))
    with open(os.path.join(output, "voice.json"), "w", encoding="utf8") as f:
        json.dump({"voice": f"Toucan embedding GAN, voice {index}", "volunteers_compared": len(volunteers),
                   "highest_similarity": round(highest, 3), "threshold": args.threshold,
                   "training": training_info(checkpoint_path),
                   "prepared_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
                  f, indent=2)
    print(f"Released to {output}: artificial voice {index}, at most {highest:.2f} similar to any of "
          f"{len(volunteers)} volunteers (threshold {args.threshold}).")


if __name__ == "__main__":
    main()
