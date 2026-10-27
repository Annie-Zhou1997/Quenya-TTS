"""
Reads Quenya text aloud: plays it (--play), saves it as a WAV file (-o), or both.

Without a Quenya checkpoint it uses the pretrained multilingual ToucanTTS model (downloaded from
Hugging Face on first use), which has never heard Quenya: the pronunciation comes from the quenya_g2p
rules, the voice and rhythm from the languages the model was trained on. After fine-tuning, pass
--model Models/ToucanTTS_Quenya/best.pt to use the Quenya model.

Examples, from the repository root:
    python -m quenya_tts.read "Elen síla lúmenn' omentielvo" -o audios/elen_sila.wav
    python -m quenya_tts.read --file chapter.txt -o audios/chapter.wav
    python -m quenya_tts.read "sinyë" --play            # only play it, no file
    python -m quenya_tts.read "Namárië" --phonemes   # only print the phonemes, no audio

Every reading carries an inaudible watermark and an "AI-generated" label (quenya_tts.watermark), as the EU AI
Act requires for synthetic audio.
"""

import argparse
import os

import numpy as np

from quenya_g2p import to_ipa


FRAME_SECONDS = 384 / 24000  # one spectrogram frame of the ToucanTTS vocoder


def synthesize(tts, phones, speed=1.0, lead_seconds=0.15, pitch_variance=1.0, creativity=0.0, watermark=True):
    """
    Speaks a Toucan phoneme string with a real pause before the first sound.

    Toucan always shortens the opening pause to a single frame, so the speech starts at once and the first
    syllable comes out clipped. Here a second pause symbol follows the opening one, the durations are
    predicted once, that second pause is set to lead_seconds, and the sentence is spoken again with those
    durations. With creativity=0 both runs predict the same pitch and energy. Whatever the model makes during
    the lead-in is then muted, so it starts in true silence.

    pitch_variance scales how far the melody moves (below 1 is flatter, more chant-like); creativity is
    Toucan's prosody sampling temperature (0 always gives the same reading, about 0.3-0.6 adds variety).

    The result carries the AI-generated watermark (quenya_tts.watermark), as the EU AI Act requires; turn it
    off only for audio that never leaves your own computer.
    """
    wave, sr = _speak(tts, phones, speed, lead_seconds, pitch_variance, creativity)
    if watermark:
        from quenya_tts.watermark import mark
        wave = mark(wave, sr)
    return wave, sr


def _speak(tts, phones, speed, lead_seconds, pitch_variance, creativity):
    prosody = dict(pitch_variance_scale=pitch_variance, prosody_creativity=creativity)
    if lead_seconds <= 0:
        return tts(phones, input_is_phones=True, duration_scaling_factor=speed, **prosody)
    phones = "~" + phones if phones.startswith("~") else "~~" + phones
    _, _, durations, _ = tts(phones, input_is_phones=True, duration_scaling_factor=speed, return_everything=True,
                             **prosody)
    durations = durations.squeeze(0).clone()
    durations[1] = max(int(durations[1]), round(lead_seconds / FRAME_SECONDS))
    wave, sr = tts(phones, input_is_phones=True, durations=durations, **prosody)
    return silence_lead(wave, sr, int(durations[0] + durations[1]), int(durations.sum())), sr


def silence_lead(wave, sr, lead_frames, total_frames, fade_seconds=0.01):
    """
    Mutes the audio the model makes during the lead-in pause, fading in just before the first sound.

    The model does not keep pauses silent: it often puts a short burst into the lead-in, which sounds and
    is recognised as an extra consonant (k, p, dʒ) before the first word.
    """
    wave = wave.copy()
    end = round(len(wave) * lead_frames / total_frames)
    fade = min(end, round(fade_seconds * sr))
    wave[:end - fade] = 0
    wave[end - fade:end] *= np.linspace(0.0, 1.0, fade, endpoint=False, dtype=wave.dtype)
    return wave


def load_reader(model=None, speaker=None, device="cpu"):
    """Returns (tts, phonemize): a ToucanTTS interface with the Quenya settings, and the Quenya phonemizer."""
    from quenya_tts.toucan import PROXY_LANGUAGE, QuenyaPhonemizer, setup_toucan
    setup_toucan()
    from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

    phonemize = QuenyaPhonemizer(device=device)
    tts = ToucanTTSInterface(device=device, tts_model_path=model, language=PROXY_LANGUAGE)
    if speaker:
        tts.set_utterance_embedding(speaker)
    return tts, phonemize


def main():
    parser = argparse.ArgumentParser(description="Read Quenya text aloud with IMS-Toucan.")
    parser.add_argument("text", nargs="?", help="Quenya text to read")
    parser.add_argument("--file", help="read the text from this file instead (one sentence or line per line)")
    parser.add_argument("-o", "--output", default=None,
                        help="where to save the WAV file (default: audios/quenya.wav, or no file with --play)")
    parser.add_argument("--play", action="store_true", help="play the audio (needs sounddevice)")
    parser.add_argument("--model", default=None, help="ToucanTTS checkpoint (.pt); default: the pretrained multilingual model")
    parser.add_argument("--speaker", default=None, help="a WAV file whose voice the model should imitate")
    parser.add_argument("--speed", type=float, default=1.0, help="duration scaling: above 1 is slower, below 1 faster")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--pitch_variance", type=float, default=1.0,
                        help="how far the melody moves: below 1 is flatter and more chant-like, above 1 livelier")
    parser.add_argument("--creativity", type=float, default=0.0,
                        help="prosody sampling temperature: 0 always reads the same way, 0.3-0.6 adds variety")
    parser.add_argument("--lead", type=float, default=0.15,
                        help="seconds of pause the model speaks before the first sound (Toucan otherwise starts "
                             "speaking at once, which clips the first syllable)")
    parser.add_argument("--padding", type=float, default=0.3,
                        help="seconds of silence at the start and end of the file (many players and Bluetooth "
                             "headphones swallow the first moments of a file, which cuts off the first syllable)")
    parser.add_argument("--phonemes", action="store_true", help="only print the phonemes and exit")
    parser.add_argument("--no-watermark", dest="watermark", action="store_false",
                        help="leave out the AI-generated watermark: only for audio that never leaves your computer, "
                             "since the EU AI Act requires it on anything shared")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf8") as f:
            lines = [line.strip() for line in f if line.strip()]
    elif args.text:
        lines = [args.text]
    else:
        parser.error("give some text or --file")

    for line in lines:
        print(f"{line}\n  {to_ipa(line)}")
    if args.phonemes:
        return

    if not args.output and not args.play:
        args.output = "audios/quenya.wav"
    output = os.path.abspath(args.output) if args.output else None
    model = os.path.abspath(args.model) if args.model else None
    speaker = os.path.abspath(args.speaker) if args.speaker else None

    tts, phonemize = load_reader(model=model, speaker=speaker, device=args.device)
    import torch
    from Utility.utils import float2pcm

    from quenya_tts.watermark import write_generated

    # the same loop as ToucanTTSInterface.read_to_file, but with phonemes as input and longer silence at both ends
    sr = 24000
    padding = torch.zeros([int(args.padding * sr)])
    silence = torch.zeros([400])
    wav = padding.clone()
    for line in lines:
        spoken, sr = synthesize(tts, phonemize(line), speed=args.speed, lead_seconds=args.lead,
                                pitch_variance=args.pitch_variance, creativity=args.creativity,
                                watermark=args.watermark)
        wav = torch.cat((wav, torch.tensor(spoken).cpu(), silence), 0)
    wav = torch.cat((wav, padding), 0)
    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        write_generated(output, float2pcm(wav), sr)
        print(f"Saved {args.output}")
    if args.play:
        import sounddevice
        sounddevice.play(wav.numpy(), sr)
        sounddevice.wait()


if __name__ == "__main__":
    main()
