# Recording a voice

A fine-tuned voice needs about 1 to 2 hours of one person reading Quenya clearly. [script.tsv](script.tsv)
holds 2,500 lines for that, about 90 minutes of speech, in 25 sessions of 100 lines.

## What is in the script

| Kind | Lines | Source |
|---|---|---|
| `book` | 81 | the recordings of *Ma Quetilyë Quenya?* (`audio/parande_N/SCRIPT.md`), first, because the book needs them anyway |
| `eldamo` | about 900 | short attested phrases, and attested words four to a line (verbs as *tirin* "I watch", *valtan* "I excite") |
| `sentence` | about 1,500 | short sentences from attested words with simple, regular grammar: *I cirya ná ninquë.* *Ma i cirya ná ninquë?* *Cenin i cirya.* *I cirya ar i elen.* The words are attested; the sentences are made up and often odd, which does not matter for training |

The `gloss` column says what each line means. After the book lines, lines are ordered by the sounds they
add, so the first sessions already cover every sound: after session 3 (about 12 minutes of speech) every
Quenya sound has at least ten examples. If you stop early, you still leave a useful voice.

Words and phrases come from [Eldamo](https://eldamo.org) by Paul Strack (CC BY 4.0), Late Quenya only,
without reconstructed or doubtful forms, in Third Age spelling. To rebuild the script (for example after
the book gets new chapters), see [build_script.py](build_script.py).

## Before you start

- **Room**: the quietest room you have, with soft things around (curtains, sofa, a wardrobe of clothes).
  No fridge, fan or traffic noise. Always the same room.
- **Microphone**: any decent USB or headset microphone, always the same one, about a hand's width from
  your mouth and a little to the side. Not the laptop's built-in microphone.
- **Level**: speak normally and set the input so it never clips. The recorder warns you when a take is
  too loud or too quiet.
- **Format**: WAV, mono, 48 kHz (the recorder does this for you).

## How to read

- Read the way you would speak it in the book: calm, clear, conversational. Not chanting, not acting.
- Pronounce it as Third Age Quenya (Appendix E): stress on the first syllable of two-syllable words; in longer
  words on the second-to-last if it is long (long vowel, diphthong, or two consonants after it), otherwise on
  the third-to-last; long vowels long, double consonants long, *r* trilled, and unstressed vowels kept clear
  with their full sound (Tolkien: in Quenya "the quality of unstressed vowels is clear and not obscured", so no
  reduced vowels as in English or Portuguese). If unsure, press `l` in the
  recorder to hear the line, or run `python -m quenya_tts.read "word" --play` (add `--phonemes` to only see the sounds). The TTS voice follows the
  same rules for stress, length and sounds, but it is not trained on Quenya yet: copy what it pronounces, not
  how its voice sounds.
- Keep the same voice, speed and distance in every session. Sessions of 100 lines (about 20 to 30 minutes)
  with breaks work better than long ones; stop when your voice gets tired.
- A slip is fine: redo the line. Leave a short silence before and after each line.
- Word lines (*Valtan, loica, oiavë, mahtan.*) are read as a list, with a small pause at each comma.

## Recording

With the reader's requirements installed:

```
python -m quenya_tts.record --session 1 -o ~/recordings
```

It shows each line with its meaning, can play the line read by the TTS first (`l`; the first time
loads the model), records while you read (Enter to start, Enter to stop), lets you play the take back or
redo it, and saves it as `s01_001.wav`, `s01_002.wav`, ... Lines already
recorded are skipped, so you can stop at any time and go on later with the same command. It keeps
recording for half a second after you press Enter, so the last sound is not cut off, and warns you when a
take is too loud, too quiet, or already sounding at its very start or still sounding at its very end (wait a
moment after Enter before you speak, and after you finish before you press Enter). Use
`--device` to pick the microphone (`python -m sounddevice` lists them).

Any other tool works too, as long as each line ends up in its own file named by its id.

### In the browser, for volunteers

Volunteers can record in their browser instead, with nothing to install, on the recording site
([quenya-voices](https://github.com/brunopacheco1/quenya-voices)): invitation and sign-in, consent, a microphone check, automatic quality checks on every
take, submissions and review. `python -m quenya_tts.volunteers download` brings the accepted recordings back here
for `prepare_corpus`.

## Training on the recordings

```
python -m quenya_tts.prepare_corpus --audio ~/recordings --script recording/script.tsv --speaker bruno --name Volunteers
python -m quenya_tts.train --gpu_id 0 --steps 5000
```

`prepare_corpus` checks every take with the phone recogniser and leaves out the ones it hears very
differently from the text (misreads, noise, a wrong file); `cuts.csv` in the corpus folder lists them.
Recordings in `Volunteers` must be shared under CC BY 4.0 or CC0, so the model trained on them can be
published.
