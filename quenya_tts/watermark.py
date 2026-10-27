"""
Marks generated speech as AI-generated, as Article 50(2) of the EU AI Act requires: an inaudible AudioSeal
watermark in the sound itself, and a label in the WAV file's metadata.

The watermark carries a 16-bit message naming this project, survives format changes and cutting reasonably
well, and is checked with AudioSeal's detector. AudioSeal (MIT) works at 16 kHz, so the watermark is made at
16 kHz and resampled to the audio's own rate. Its models come from Hugging Face on first use.

    pip install audioseal
    python -m quenya_tts.watermark check audios/quenya.wav    # is this file marked, and by Quenya-TTS?

quenya_tts.read.synthesize marks everything it produces, so every tool that speaks (read, record, compare,
volunteers references) is covered.
"""

import argparse

import numpy as np

AUDIOSEAL_RATE = 16000
# The 16-bit message: "QT", for Quenya-TTS
MESSAGE = 0x5154
LABEL = ("AI-generated speech made with Quenya-TTS (https://github.com/brunopacheco1/Quenya-TTS). It carries an "
         "AudioSeal watermark; check with: python -m quenya_tts.watermark check <file>")

_models = {}


def to_bits(value, bits=16):
    """The binary digits of value, most significant first."""
    return [(value >> (bits - 1 - i)) & 1 for i in range(bits)]


def from_bits(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _model(kind):
    if kind not in _models:
        from audioseal import AudioSeal
        _models[kind] = (AudioSeal.load_generator("audioseal_wm_16bits") if kind == "generator"
                         else AudioSeal.load_detector("audioseal_detector_16bits"))
    return _models[kind]


def _no_compile():
    """
    AudioSeal 0.2 runs its models through torch.compile, which needs a C++ toolchain and breaks on some setups
    (macOS with Homebrew Python, among others). Marking a few seconds of speech does not need it.
    """
    try:
        from audioseal.libs.moshi.utils.compile import no_compile
        return no_compile()
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def _resample(x, rate, target):
    import torchaudio
    return x if rate == target else torchaudio.functional.resample(x, rate, target)


def fit_length(signal, length):
    """Cuts or zero-pads the last axis of a numpy array to length samples."""
    if signal.shape[-1] >= length:
        return signal[..., :length]
    return np.concatenate([signal, np.zeros(signal.shape[:-1] + (length - signal.shape[-1],), signal.dtype)], -1)


def mark(wave, sr):
    """Returns the mono float audio wave (sampled at sr) with the Quenya-TTS watermark added."""
    import torch
    x = torch.as_tensor(np.asarray(wave, dtype=np.float32)).reshape(1, 1, -1)
    message = torch.tensor([to_bits(MESSAGE)])
    with torch.no_grad(), _no_compile():
        watermark = _model("generator").get_watermark(_resample(x, sr, AUDIOSEAL_RATE), message=message)
        watermark = _resample(watermark, AUDIOSEAL_RATE, sr)
    return np.asarray(wave, dtype=np.float32) + fit_length(watermark.reshape(-1).numpy(), x.shape[-1])


def detect(wave, sr):
    """Returns (probability that the audio is watermarked, whether the message is Quenya-TTS's)."""
    import torch
    x = torch.as_tensor(np.asarray(wave, dtype=np.float32)).reshape(1, 1, -1)
    with torch.no_grad(), _no_compile():
        probability, message = _model("detector").detect_watermark(_resample(x, sr, AUDIOSEAL_RATE))
    return float(probability[0]), from_bits(message[0].tolist()) == MESSAGE


def write_generated(path, wave, sr, subtype="PCM_16"):
    """Saves generated (already watermarked) audio as WAV, labelled as AI-generated in its metadata."""
    import soundfile
    with soundfile.SoundFile(path, "w", samplerate=sr, channels=1, subtype=subtype, format="WAV") as f:
        f.software = "Quenya-TTS"
        f.comment = LABEL
        f.write(wave)


def main():
    parser = argparse.ArgumentParser(description="Check audio files for the Quenya-TTS watermark.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="say whether each file carries the watermark")
    check.add_argument("files", nargs="+")
    args = parser.parse_args()

    import soundfile
    for path in args.files:
        wave, sr = soundfile.read(path, dtype="float32", always_2d=True)
        probability, ours = detect(wave.mean(axis=1), sr)
        if probability < 0.5:
            verdict = "no watermark found"
        else:
            verdict = "watermarked by Quenya-TTS" if ours else "watermarked, but not with the Quenya-TTS message"
        print(f"{path}: {verdict} (detector: {probability:.0%})")


if __name__ == "__main__":
    main()
