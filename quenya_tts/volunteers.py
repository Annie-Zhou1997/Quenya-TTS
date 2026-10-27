"""
Connects the volunteer recording site (https://github.com/brunopacheco1/quenya-voices) with training: downloads the accepted recordings, and
uploads the reference readings that the site's Listen button plays.

The site keeps everything in Gcore object storage in Luxembourg, encrypted object by object with the site's key
(see the quenya-voices README). This script reads it through the S3 API and decrypts it, with the same settings
as the site (/etc/quenya-voices.env on its server):

    export STORAGE_ENDPOINT=https://luxembourg-2.storage.gcore.dev  STORAGE_REGION=s-ed1
    export STORAGE_ACCESS_KEY_ID=...  STORAGE_SECRET_ACCESS_KEY=...  STORAGE_KEY=...
    pip install boto3 cryptography

Download what reviewers accepted (only the lines they marked good), then build the corpus:

    python -m quenya_tts.volunteers download -o ~/volunteers
    python -m quenya_tts.volunteers download -o ~/volunteers --for-training   # refuses until training is allowed

Each accepted recording lands in its own folder, <volunteer>/<recording>/<line id>.wav, and the command prints the
prepare_corpus call for each. volunteers.tsv says how to credit each volunteer. Volunteers who asked for erasure are deleted from the download folder on every run, so run it before
every training.

--for-training enforces the promise on the site's Privacy Notice: no training before the accepted speech of
all volunteers reaches the minimum (minimum_training_minutes in the site's site.json), and none in the waiting
period after it first did (training_wait_days). The site records both in stats/milestones.json when the minimum
is reached; the defaults below are only used to report progress before that.

Upload reference readings for Listen (needs the reader's requirements):

    python -m quenya_tts.volunteers references --session 1 -o ~/references --upload
"""

import argparse
import base64
import csv
import datetime
import json
import os
import shlex
import shutil

# The site's defaults (site.json in quenya-voices). The values the site saved in stats/milestones.json win.
SITE = {"minimum_training_minutes": 60, "training_wait_days": 30}
SCRIPT_TSV = os.path.join(os.path.dirname(__file__), "..", "recording", "script.tsv")


# The site's encryption (quenya-voices src/storage.js): "QVE1", a 12-byte IV, then AES-256-GCM ciphertext and tag.
# Metadata is sealed the same way into the object's one metadata value, "sealed" (base64).
MAGIC = b"QVE1"
OVERHEAD = len(MAGIC) + 12 + 16


def seal(key, data):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    iv = os.urandom(12)
    return MAGIC + iv + AESGCM(key).encrypt(iv, data, None)


def unseal(key, data):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if not data.startswith(MAGIC) or len(data) < OVERHEAD:
        raise ValueError("not an object encrypted by the site")
    return AESGCM(key).decrypt(data[4:16], data[16:], None)


def storage_key(text):
    key = base64.b64decode(text or "")
    if len(key) != 32:
        raise ValueError("STORAGE_KEY must be the site's 32-byte key, base64-encoded")
    return key


class Bucket:
    """The few storage operations this script needs, over the S3 API, decrypting and encrypting like the site."""

    def __init__(self, name, client, key):
        self.name, self.client, self.key = name, client, key

    @classmethod
    def from_environment(cls, name):
        import boto3
        client = boto3.client("s3", endpoint_url=os.environ["STORAGE_ENDPOINT"],
                              region_name=os.environ["STORAGE_REGION"],
                              aws_access_key_id=os.environ["STORAGE_ACCESS_KEY_ID"],
                              aws_secret_access_key=os.environ["STORAGE_SECRET_ACCESS_KEY"])
        return cls(name, client, storage_key(os.environ.get("STORAGE_KEY")))

    def keys(self, prefix):
        """Returns [(key, size)] under prefix; sizes are those of the decrypted objects."""
        found = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.name, Prefix=prefix):
            found.extend((item["Key"], item["Size"] - OVERHEAD) for item in page.get("Contents", []))
        return found

    def read(self, key):
        return unseal(self.key, self.client.get_object(Bucket=self.name, Key=key)["Body"].read())

    def read_json(self, key):
        try:
            return json.loads(self.read(key))
        except self.client.exceptions.NoSuchKey:
            return None

    def metadata(self, key):
        sealed = self.client.head_object(Bucket=self.name, Key=key).get("Metadata", {}).get("sealed")
        return json.loads(unseal(self.key, base64.b64decode(sealed))) if sealed else {}

    def write(self, key, data, content_type):
        sealed = base64.b64encode(seal(self.key, b"{}")).decode()
        self.client.put_object(Bucket=self.name, Key=key, Body=seal(self.key, data),
                               ContentType="application/octet-stream", Metadata={"sealed": sealed})


def parse_time(text):
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))


def accepted_seconds(bucket):
    """Accepted speech per volunteer, from the site's accepted/ markers (their metadata holds the seconds)."""
    totals = {}
    for key, _ in bucket.keys("accepted/"):
        volunteer = key.split("/")[1]
        totals[volunteer] = totals.get(volunteer, 0.0) + float(bucket.metadata(key).get("seconds", 0))
    return totals


def training_gate(total_seconds, milestones, now, defaults=SITE):
    """Returns None if training is allowed now, otherwise why not."""
    milestones = milestones or {}
    minimum = milestones.get("minimum_minutes", defaults["minimum_training_minutes"])
    if total_seconds < minimum * 60:
        return f"only {total_seconds / 60:.1f} of the {minimum} minutes of accepted speech needed before any training"
    reached = milestones.get("minimum_reached_at")
    if not reached:
        return "the site has not recorded when the minimum was reached yet (it does so when a reviewer accepts a recording)"
    wait = milestones.get("training_wait_days", defaults["training_wait_days"])
    earliest = parse_time(reached) + datetime.timedelta(days=wait)
    if now < earliest:
        return f"the waiting period after the minimum was reached ends on {earliest.date().isoformat()}"
    return None


def remove_erased(bucket, output):
    """Deletes local copies of recordings made before each erasure. Returns the erased volunteer ids."""
    erased = set()
    for key, _ in bucket.keys("erasures/"):
        record = bucket.read_json(key)
        if not record:
            continue
        volunteer, erased_at = record["volunteer"], parse_time(record["erased_at"])
        erased.add(volunteer)
        folder = os.path.join(output, volunteer)
        if not os.path.isdir(folder):
            continue
        for recording_id in os.listdir(folder):
            meta_path = os.path.join(folder, recording_id, "recording.json")
            created = None
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf8") as f:
                    created = parse_time(json.load(f)["created_at"])
            if created is None or created < erased_at:
                shutil.rmtree(os.path.join(folder, recording_id))
                print(f"deleted {volunteer}/{recording_id}: the volunteer asked for erasure")
        if not os.listdir(folder):
            os.rmdir(folder)
    return erased


def speaker_name(volunteer, recording_id):
    return f"v{volunteer[:8]}_{recording_id}"


def download(bucket, output, script=SCRIPT_TSV):
    """Downloads every accepted recording's good lines. Returns [(folder, speaker, consent, recording, lines)]."""
    os.makedirs(output, exist_ok=True)
    remove_erased(bucket, output)
    downloaded = []
    for key, _ in sorted(bucket.keys("accepted/")):
        _, volunteer, recording_id = key.split("/")
        consent = bucket.read_json(f"volunteers/{volunteer}/consent.json")
        recording = bucket.read_json(f"recordings/{volunteer}/{recording_id}/recording.json")
        if not consent or not recording or recording.get("status") != "accepted":
            continue  # erased since, or not accepted any more
        good = sorted(line for line, review in recording.get("review", {}).items() if review.get("status") == "good")
        folder = os.path.join(output, volunteer, recording_id)
        os.makedirs(folder, exist_ok=True)
        sizes = dict(bucket.keys(f"recordings/{volunteer}/{recording_id}/"))
        for line in good:
            remote = f"recordings/{volunteer}/{recording_id}/{line}.wav"
            local = os.path.join(folder, f"{line}.wav")
            if remote not in sizes:
                continue
            if not (os.path.exists(local) and os.path.getsize(local) == sizes[remote]):
                with open(local, "wb") as f:
                    f.write(bucket.read(remote))
        with open(os.path.join(folder, "recording.json"), "w", encoding="utf8") as f:
            json.dump(recording, f, indent=2, ensure_ascii=False)
        downloaded.append((folder, speaker_name(volunteer, recording_id), consent, recording, good))
    write_volunteers(output, downloaded)
    return downloaded


def write_volunteers(output, downloaded):
    """volunteers.tsv: one row per accepted recording, with how to credit the volunteer and how they share."""
    with open(os.path.join(output, "volunteers.tsv"), "w", encoding="utf8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["folder", "speaker", "credit_name", "sharing", "consent_version", "session", "good_lines"])
        for folder, speaker, consent, recording, good in downloaded:
            writer.writerow([os.path.relpath(folder, output), speaker, consent.get("credit_name") or "",
                             consent["licence"], consent["consent_version"], recording["session"], len(good)])


def references(args):
    from quenya_tts.read import load_reader, synthesize
    from quenya_tts.record import session_lines
    from quenya_tts.watermark import write_generated

    lines = session_lines(args.script, args.session)
    os.makedirs(args.output, exist_ok=True)
    tts, phonemize = load_reader(model=args.model, device=args.device)
    bucket = Bucket.from_environment(args.bucket) if args.upload else None
    for line_id, text, _ in lines:
        path = os.path.join(args.output, f"{line_id}.wav")
        if not os.path.exists(path):
            wave, sr = synthesize(tts, phonemize(text))
            write_generated(path, wave, sr)
        if bucket:
            with open(path, "rb") as f:
                bucket.write(f"reference/{line_id}.wav", f.read(), "audio/wav")
        print(f"{line_id}: {text}")


def main():
    parser = argparse.ArgumentParser(description="Download accepted volunteer recordings, or upload reference readings.")
    parser.add_argument("--bucket", default=os.environ.get("STORAGE_BUCKET", "quenya-recordings"))
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("download", help="download the accepted recordings")
    get.add_argument("-o", "--output", required=True, help="download folder, kept between runs")
    get.add_argument("--for-training", action="store_true", help="refuse unless training is allowed now")
    get.add_argument("--script", default=SCRIPT_TSV)
    ref = commands.add_parser("references", help="read a session with the TTS, for the site's Listen button")
    ref.add_argument("--session", type=int, required=True)
    ref.add_argument("-o", "--output", required=True)
    ref.add_argument("--upload", action="store_true", help="also upload them to the bucket")
    ref.add_argument("--script", default=SCRIPT_TSV)
    ref.add_argument("--model", default=None, help="ToucanTTS checkpoint (default: the pretrained model)")
    ref.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.command == "references":
        references(args)
        return
    bucket = Bucket.from_environment(args.bucket)
    if args.for_training:
        total = sum(accepted_seconds(bucket).values())
        reason = training_gate(total, bucket.read_json("stats/milestones.json"),
                               datetime.datetime.now(datetime.timezone.utc))
        if reason:
            parser.exit(1, f"Training is not allowed yet: {reason}.\n")
    downloaded = download(bucket, args.output, args.script)
    for folder, speaker, _, recording, good in downloaded:
        # quoted, so that the printed command is safe to run whatever the names in the bucket
        print(shlex.join(["python", "-m", "quenya_tts.prepare_corpus", "--audio", str(folder), "--script",
                          str(args.script), "--speaker", str(speaker), "--name", "Volunteers"]))
    print(f"{len(downloaded)} accepted recordings in {args.output} (credits: volunteers.tsv)")


if __name__ == "__main__":
    main()
