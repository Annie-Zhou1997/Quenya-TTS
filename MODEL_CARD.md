# Quenya-TTS model card

A text-to-speech model that reads Quenya, Tolkien's High-elven, with Third Age pronunciation (*The Lord of the
Rings*, Appendix E). It is IMS-Toucan's multilingual ToucanTTS model, fine-tuned on recordings of volunteers.
`python -m quenya_tts.publish` prepares every release and copies this card into it, together with
[MODEL_LICENSE.md](MODEL_LICENSE.md) and `voice.json`.

## Voice

The model speaks with an **artificial voice**: a speaker embedding made by Toucan's voice generator, for a voice
that does not exist. Before each release `quenya_tts.publish` checks that this voice does not sound like any
volunteer whose recordings were used. Its speaker embedding must be less similar to each volunteer's than
SpeechBrain's own same-speaker threshold (cosine 0.25). `voice.json` in the release records the voice, the
highest similarity found and how many volunteers were compared.

A model still learns from every voice it hears. Given a sample of someone's voice (`--speaker`), it can imitate
that person, as the pretrained Toucan model already can. The licence forbids doing so without their consent.

## Training data

- **Volunteer recordings** made on the [Quenya Voices](https://github.com/brunopacheco1/quenya-voices) site.
  Volunteers are invited, are 18 or older, record only their own voice, and consent to their recordings being
  used to train openly published models. A reviewer accepts each recording before it is used. The recordings
  themselves are never published, and a volunteer's erasure removes their recordings from every later training.
  Models already released stay as they are.
- Nothing else. `quenya_tts.publish` refuses a checkpoint whose `training.json` lists other corpora. The thesis
  recordings (Atanquesta, Glǽmscrafu) are for experiments only.

Training starts only once the accepted recordings of all volunteers reach the minimum set on the site, and the
first training waits 30 days after that.

## Intended use

Reading Quenya aloud: audio for the book *Ma Quetilyë Quenya?*, learners' pronunciation practice, and other
Quenya projects.

## Uses that are not allowed

See the use restrictions in [MODEL_LICENSE.md](MODEL_LICENSE.md). In short: no imitating real people without
their consent, no deceiving anyone about whether speech is real or who is speaking, no removing or hiding the
watermark, and nothing unlawful.

## Marking: the EU AI Act

Every file the tools produce (`quenya_tts.read`, `record`, `compare`, `volunteers references`) carries an
inaudible [AudioSeal](https://github.com/facebookresearch/audioseal) watermark with the 16-bit message `QT` (0x5154).
WAV files also say in their metadata that they are AI-generated. This is the machine-readable marking that
Article 50(2) of the EU AI Act (Regulation (EU) 2024/1689) asks of providers of systems that generate synthetic
audio. To check a file:

```
python -m quenya_tts.watermark check file.wav
```

Whoever publishes audio made with this model should also say, where people hear it, that it is AI-generated.

## Limitations

- The model learns from a few hours of speech at most: expect wrong rhythms, slips between similar sounds (r and
  l, d), and a melody that is sometimes too high or flat.
- Quenya borrows Toucan's Finnish language embedding, and the pronunciation comes from the rules in `quenya_g2p`.
  Where those rules differ from your reading of Tolkien, the model follows the rules.
- It reads Quenya only. Text in other languages comes out with Quenya rules.

## Licence and credits

The model weights are released under the licence in [MODEL_LICENSE.md](MODEL_LICENSE.md). The code is Apache-2.0
([LICENSE](LICENSE)). The volunteers who wish to be credited are listed in the release's credits, by the name
they gave on the site. IMS-Toucan is by the Institute for Natural Language Processing (IMS), University of
Stuttgart. The original Quenya-TTS is by Annie Zhou (University of Groningen, 2024).
