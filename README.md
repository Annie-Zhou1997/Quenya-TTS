# 🧝🏽‍♀️ Quenya TTS

This project is a Quenya Text-to-Speech (TTS) system created using [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan). It is part of a master's thesis project for the MSc in Voice Technology at the [University of Groningen](https://www.rug.nl/). The demo audio can be found [here](https://annie-zhou1997.github.io/QuenyaTTS.github.io/).

This study evaluates the use of articulatory features as inputs for speech synthesis and examines the outcomes after applying transfer learning from models based on more resourced languages. The project utilizes the ToucanTTS system from the University of Stuttgart, which is based on the FastSpeech2 architecture and converts text into articulatory features for speech synthesis. A TTS system for Quenya was developed by fine-tuning three models using a 34-minute Quenya dataset: one for Finnish, one for English, and a multilingual model provided by ToucanTTS. The results showed that the Finnish fine-tuned model produced better speech quality than the English fine-tuned model, while the multilingual fine-tuned model produced the most natural and accurate speech.


# Acknowledgements
I would like to express my deepest gratitude to my supervisor, Ph.D. Candidate Phat Do, and Associate Professor Dr. Matt Coler, for their invaluable guidance throughout this project. My thanks also go to the original creators of the [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan) system for providing such a convenient tool. Special thanks to the author of the book [Atanquesta](https://middangeard.org.uk/aglardh/atanquesta), whose open-source recordings were crucial to this project. I am also deeply grateful to [Glaemscrafu](https://glaemscrafu.jrrvf.com/english/index.html), a website rich with resources, particularly the recordings by Benjamin Babut and Bertrand Bellet. Additionally, I extend my gratitude to all the participants in the questionnaire survey and the Quenya enthusiasts who were willing to write detailed listening reports. Finally, I am deeply grateful to the volunteer speaker, Igor Marchenko. Despite your busy schedule leading up to graduation, you provided exceptionally high-quality recordings. You are the best classmate I have ever had.

# Experiment

For details on environment configuration, please refer to the [original repository](https://github.com/DigitalPhonetics/IMS-Toucan).

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

