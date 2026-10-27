#!/usr/bin/env bash
# Runs one training on a fresh GPU machine: installs Quenya-TTS, fetches the accepted volunteer recordings from the site's storage,
# trains, and writes the release (artificial voice, watermarked samples) to $RUN/out. quenya_tts.gcore sends it
# over SSH, after exports for the variables below; the storage keys therefore never touch the machine's disk. Everything
# the run writes, recordings and checkpoints included, is deleted with the machine.
#
#   BRANCH          Quenya-TTS branch to train with
#   STEPS           training steps
#   BATCH_SIZE      training batch size
#   TRAIN_SECONDS   stop training after this long, so the results can still be copied back before the time limit
#   KEEP_CHECKPOINT 1 copies best.pt to out/private (it holds a volunteer's voice: keep it private)
#   STORAGE_ENDPOINT, STORAGE_REGION, STORAGE_ACCESS_KEY_ID, STORAGE_SECRET_ACCESS_KEY, STORAGE_KEY
set -euo pipefail

RUN="$HOME/run"
REPO="$RUN/Quenya-TTS"
PY="$RUN/py311/bin/python"
OUT="$RUN/out"
export QUENYA_CORPORA_DIR="$RUN/corpora" MPLBACKEND=Agg
mkdir -p "$OUT"

echo "== Installing"
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update > /dev/null
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y git espeak-ng libportaudio2 ffmpeg python3-venv > /dev/null
git clone -q --branch "$BRANCH" --recurse-submodules https://github.com/brunopacheco1/Quenya-TTS.git "$REPO"
python3 -m venv "$RUN/tools" && "$RUN/tools/bin/pip" install -q uv
"$RUN/tools/bin/uv" venv -q --seed --python 3.11 "$RUN/py311"
cd "$REPO"
grep -viE '^(pyqt5|pyqtgraph|pynput|PyAutoGUI|gradio)' requirements.txt > "$RUN/requirements.txt"
"$RUN/tools/bin/uv" pip install -q --python "$PY" -r "$RUN/requirements.txt" "transformers>=4.46,<4.50" boto3 cryptography audioseal
"$PY" -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.__version__, torch.cuda.get_device_name(0))"

echo "== Fetching the accepted recordings"
"$PY" -m quenya_tts.volunteers download -o "$RUN/volunteers" --for-training | tee "$OUT/download.log"
# The downloader prints one quoted prepare_corpus command per recording: run each as an argument list, never
# through a shell, since the paths come from names in the bucket.
"$PY" - "$OUT/download.log" "$PY" <<'EOF'
import shlex, subprocess, sys
commands = [line for line in open(sys.argv[1]) if line.startswith("python -m quenya_tts.prepare_corpus")]
if not commands:
    sys.exit("nothing to train on (see the message above)")
for command in commands:
    subprocess.run([sys.argv[2], *shlex.split(command)[1:], "--device", "cuda"], check=True)
EOF

echo "== Training ($STEPS steps, at most $((TRAIN_SECONDS / 60)) min)"
# Toucan saves best.pt every 250 steps, so a run stopped by the time limit still has a model.
timeout --signal=INT "$TRAIN_SECONDS" "$PY" -m quenya_tts.train --gpu_id 0 --corpora Volunteers \
    --steps "$STEPS" --batch_size "$BATCH_SIZE" 2>&1 | grep -v Warning | tee "$OUT/train.log" || true
BEST=$(ls Models/ToucanTTS_Quenya*/best.pt | tail -1)

echo "== Releasing with an artificial voice"
"$PY" -m quenya_tts.publish "$BEST" --voices "$RUN/volunteers" -o "$OUT/release" --device cuda
"$PY" -m quenya_tts.read "Elen síla lúmenn' omentielvo." --model "$OUT/release/quenya_tts.pt" --device cuda \
    -o "$OUT/elen_sila.wav" 2>/dev/null
"$PY" -m quenya_tts.read "Ma quetilyë Quenya? Quetin Quenya." --model "$OUT/release/quenya_tts.pt" --device cuda \
    -o "$OUT/dialogue.wav" 2>/dev/null
"$PY" -m quenya_tts.watermark check "$OUT/elen_sila.wav" "$OUT/dialogue.wav" 2>/dev/null | tee "$OUT/watermark.txt"
if [ "${KEEP_CHECKPOINT:-0}" = 1 ]; then
    mkdir -p "$OUT/private" && cp "$BEST" "$(dirname "$BEST")/training.json" "$OUT/private/"
fi
echo "== Done"
