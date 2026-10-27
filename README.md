# Quenya-TTS

Text-to-speech for Quenya, built on [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan). It started as Annie Zhou's master's thesis at the University of Groningen (described below) and is being continued to produce audio for the community book [Ma Quetelyë Quenya?](https://ma-quetelye-quenya.tolkien.lu/).

## How it fits together

Toucan is used as it is, without changes. Quenya text is turned into IPA by our own rules and handed to Toucan as phonemes, which Toucan accepts for reading and for training. Toucan has no Quenya language embedding, so Quenya borrows the Finnish one.

- **[quenya_g2p/](quenya_g2p)** turns Quenya text into IPA. It follows Third Age Quenya as described in Appendix E of The Lord of the Rings; the rules, and the choices made where sources disagree, are documented in [quenya_g2p/rules.py](quenya_g2p/rules.py). It is a small Python package with no dependencies, so other projects can use it too. Tests are in [tests/](tests).
- **[quenya_tts/](quenya_tts)** connects it to Toucan: a reader that turns Quenya text into a WAV file, a training script, and the corpus loaders.
- **[toucan/](toucan)** is IMS-Toucan as a git submodule, pinned to a commit of its `MassiveScaleToucan` branch. Updating Toucan means moving the submodule to a newer commit.

There is no trained Quenya checkpoint yet. The 2024 thesis code, which was a modified copy of an older Toucan, is in this repository's git history before the move to this layout.

## Quick start

Quenya to IPA needs only Python:

```
pip install .            # or: pip install git+https://github.com/brunopacheco1/Quenya-TTS
python -m quenya_g2p "Elen síla lúmenn' omentielvo"
python -m unittest discover tests
```

To hear it, get Toucan and its requirements. Use Python 3.10 or 3.11, because some of Toucan's pinned packages do not install on newer versions. Toucan also needs espeak-ng (`brew install espeak-ng` on macOS, `apt install espeak-ng libportaudio2` on Debian or Ubuntu).

```
git submodule update --init
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m quenya_tts.read "Elen síla lúmenn' omentielvo" -o audios/elen_sila.wav
```

Add `--play` to hear it; with `--play` and no `-o` it only plays and saves no file: `python -m quenya_tts.read "sinyë" --play`.

On macOS, if phonemizer cannot find espeak-ng, set `PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib`.

The reader uses the pretrained multilingual Toucan model, downloaded from Hugging Face on first use. That model has never heard Quenya, so the pronunciation follows our rules but the voice and rhythm are generic.

## Comparing with a recording

`quenya_tts.compare` checks our output against a reference recording of the same text, such as a reading of Namárië. For each line it lists what a phone recogniser hears in both recordings next to the IPA from our rules, plots both pitch contours aligned in time, and saves both versions of the line as WAV files. It needs one more package:

```
pip install "transformers>=4.46,<4.50"
python -m quenya_tts.compare path/to/recording.mp3 --text namarie.txt -o comparisons/namarie
```

The text file has one line per spoken line. The recording is cut into lines where the recogniser hears each line's sounds, so lines the reader runs together are still found. Check the cuts with `--segments_only`; if one is wrong, fix the times in `segments.csv` and run again with `--segments comparisons/namarie/segments.csv`. Keep recordings you have no rights to outside the repository.

## Training

Each corpus is a folder with `transcript.txt` (one `file_name|Quenya text` per line) next to its audio files, under `Corpora/Quenya/<name>` (or `$QUENYA_CORPORA_DIR/<name>`):

- `Volunteers`: recordings made by community volunteers on the [recording site](https://github.com/brunopacheco1/quenya-voices), accepted by a reviewer, with each volunteer's consent to training (their recordings stay private). The public model is trained only on these. `python -m quenya_tts.volunteers download` fetches them.
- `Atanquesta`, `Glaemscrafu`: the recordings used in the thesis. Atanquesta is CC BY-NC-SA 4.0, so models trained on these are for experiments only. Pass `--experimental` to include them.

Recordings of whole texts (a poem, a prayer) can be turned into a corpus with `quenya_tts.prepare_corpus`. It pairs `<name>.mp3` with `<name>.txt` (one line of Quenya per line), cuts each recording into one clip per line with the comparison tool's phone recogniser, leaves out lines it hears too differently from the text, and writes the clips and `transcript.txt`:

```
python -m quenya_tts.prepare_corpus --audio recordings/ --texts texts/ --name Glaemscrafu
```

To record a new voice, [recording/](recording) has a 2,500-line script (about 90 minutes of speech), a recorder that saves one file per line (`python -m quenya_tts.record`), and a guide on how to read. `prepare_corpus --script recording/script.tsv --speaker <name>` turns those files into a corpus. Volunteers record in their browser on the [recording site](https://github.com/brunopacheco1/quenya-voices) instead.

Training needs a CUDA GPU:

```
python -m quenya_tts.train --gpu_id 0
python -m quenya_tts.train --gpu_id 0 --corpora Glaemscrafu --steps 500   # a short test run on one corpus
python -m quenya_tts.read "Namárië" --model Models/ToucanTTS_Quenya/best.pt -o audios/namarie.wav
```

### Training on volunteer recordings: Gcore

The public model is trained on a Gcore GPU machine in the EU, which `quenya_tts.gcore` creates for one run and deletes afterwards, with its disk and IP, whether training worked, failed or was stopped. The volunteer recordings go straight from the recording site's storage to that machine, so they are deleted with it and fetched again for every run: volunteers who asked for erasure are left out automatically. Only the release (artificial voice), logs and watermarked samples come back.

```
pip install gcore boto3 cryptography
export GCORE_API_KEY=...  GCORE_CLOUD_PROJECT_ID=...  GCORE_CLOUD_REGION_ID=...   # an EU region
export STORAGE_ENDPOINT=...  STORAGE_REGION=...  STORAGE_ACCESS_KEY_ID=...  STORAGE_SECRET_ACCESS_KEY=...  STORAGE_KEY=...
                                                         # the recording site's storage settings
python -m quenya_tts.gcore flavors                                                # GPU machines and prices
python -m quenya_tts.gcore train --ssh-key-name my-mac -o ~/quenya-runs --max-hours 6
python -m quenya_tts.gcore cleanup                                                # anything left behind?
```

The script refuses regions outside the EU, as the volunteers were promised. Every run has a hard time limit (`--max-hours`, 6 by default; about €17 at most on one H100). At the limit the script stops the run and deletes the machine. The machine also powers itself off at the limit, even if your computer sleeps or loses its connection; with `--watchdog` it deletes itself too, using a second API key (`GCORE_WATCHDOG_API_KEY`, give it an expiry date) that is kept only on the machine. `cleanup` lists any Quenya-TTS machine still there and deletes it with `--yes`. Before the first run: add your SSH key in Gcore (Cloud > SSH keys), check that the default security group allows SSH (port 22), and sign Gcore's data processing agreement.

### Development: Kaggle

[notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb) trains on Kaggle's free GPU, for development and experiments on recordings you may use yourself (a Kaggle dataset). Volunteer recordings never go to Kaggle.

Training checkpoints are private: Toucan stores the voice of a training clip in them as the default voice. Never share `best.pt` or any other checkpoint; release models only with `quenya_tts.publish`.

## Releasing a model

Volunteers record for training only, and are promised that a released model does not speak with their voice:

```
python -m quenya_tts.volunteers download -o ~/volunteers
python -m quenya_tts.publish Models/ToucanTTS_Quenya/best.pt --voices ~/volunteers -o release/
```

`publish` refuses a model trained on anything but the volunteer recordings, gives it an artificial voice from Toucan's voice generator, checks that this voice does not sound like any volunteer, and writes the model with the [model card](MODEL_CARD.md), the [model licence](MODEL_LICENSE.md) (Open RAIL-M with use restrictions: no imitating real people) and the volunteers' credits.

Everything the reader produces carries an inaudible AudioSeal watermark and an "AI-generated" label in the WAV metadata, as Article 50 of the EU AI Act requires for synthetic audio. Check a file with `python -m quenya_tts.watermark check file.wav`. `--no-watermark` exists only for audio that never leaves your computer.

## Master's thesis (2024)

This project is a Quenya Text-to-Speech (TTS) system created using [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan). It is part of a master's thesis project for the MSc in Voice Technology at the [University of Groningen](https://www.rug.nl/). The demo audio can be found [here](https://annie-zhou1997.github.io/QuenyaTTS.github.io/).

This study evaluates the use of articulatory features as inputs for speech synthesis and examines the outcomes after applying transfer learning from models based on more resourced languages. The project utilizes the ToucanTTS system from the University of Stuttgart, which is based on the FastSpeech2 architecture and converts text into articulatory features for speech synthesis. A TTS system for Quenya was developed by fine-tuning three models using a 34-minute Quenya dataset: one for Finnish, one for English, and a multilingual model provided by ToucanTTS. The results showed that the Finnish fine-tuned model produced better speech quality than the English fine-tuned model, while the multilingual fine-tuned model produced the most natural and accurate speech.


### Acknowledgements
I would like to express my deepest gratitude to my supervisor, Ph.D. Candidate Phat Do, and Associate Professor Dr. Matt Coler, for their invaluable guidance throughout this project. My thanks also go to the original creators of the [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan) system for providing such a convenient tool. Special thanks to the author of the book [Atanquesta](https://middangeard.org.uk/aglardh/atanquesta), whose open-source recordings were crucial to this project. I am also deeply grateful to [Glaemscrafu](https://glaemscrafu.jrrvf.com/english/index.html), a website rich with resources, particularly the recordings by Benjamin Babut and Bertrand Bellet. Additionally, I extend my gratitude to all the participants in the questionnaire survey and the Quenya enthusiasts who were willing to write detailed listening reports. Finally, I am deeply grateful to the volunteer speaker, Igor Marchenko. Despite your busy schedule leading up to graduation, you provided exceptionally high-quality recordings. You are the best classmate I have ever had.

### Experiment

_The thesis worked in a modified copy of an older Toucan, so the paths below refer to that copy, which is in the git history._

Based on the tutorial from the original repository, I have added Quenya language (qya) support in the `Preprocessing/TextFrontend.py` file and created the Quenya language G2P script, which is also located in the `Preprocessing` directory. The specific steps are as follows:

1. **Adding Quenya Language Support**:
    - Added support for the Quenya language in the `Preprocessing/TextFrontend.py` file.
    - Ensured the new language code (qya) correctly handles the special characters and phonetic rules of the Quenya language.

2. **Creating Quenya G2P Script**:
    - Developed a dedicated Quenya G2P script in the `Preprocessing` directory.
    - The script utilizes knowledge of Quenya phonology to convert text into phonemes and annotate stress.
    - It also considers some obsolete letters from early Quenya variants.

3. **Data Preprocessing**:
    - Manually segmented audio files using Audacity, ensuring each file is between 2 to 13 seconds long with accurate transcription texts.
    - Performed noise reduction on public recordings from Glæmscrafu using voicefixer.

4. **Model Training**:
    - Trained Finnish and English TTS models from scratch using the ToucanTTS system.
    - Used identical training configurations: batch size of 12, learning rate of 1e-3, and 80,000 training steps.
    - After training, generated 10 sentences not present in the training set and tested the word error rate (WER) using OpenAI’s Whisper.

5. **Fine-Tuning**:
    - Integrated the Quenya G2P script into the text front end of ToucanTTS and performed fine-tuning.
    - Fine-tuned the Finnish and English models using a learning rate of 1e-5, batch size of 6, and 6000 training steps.
    - Further fine-tuned using a multilingual pre-trained checkpoint developed with data from 12 languages: English, German, Spanish, Greek, Finnish, French, Russian, Hungarian, Dutch, Polish, Portuguese, and Italian, totaling 389 hours.
    - Averaged and consolidated the final three checkpoints into a single optimized checkpoint for inference.

6. **Evaluation**:
    - Assessed the quality of the synthesized speech using traditional Mean Opinion Score (MOS) evaluations and detailed listening reports.
    - Distributed surveys to Quenya enthusiasts and linguists, and invited two proficient Quenya speakers to identify specific errors in the generated sentences and provide detailed listening reports.
    - In the MOS evaluation, participants were provided with 9 sentences in 4 different versions: real human voice recordings, sentences fine-tuned from a multilingual model, sentences fine-tuned from Finnish, and sentences fine-tuned from English. Participants were not informed which voices were produced by humans and rated each sample on a scale from 1 to 5.
