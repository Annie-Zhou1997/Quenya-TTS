# 🧝🏽‍♀️ Quenya TTS

This project is a Quenya Text-to-Speech (TTS) system created using [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan). It is part of a master's thesis project for the MSc in Voice Technology at the [University of Groningen](https://www.rug.nl/). The demo audio can be found [here](https://annie-zhou1997.github.io/QuenyaTTS.github.io/).

This study evaluates the use of articulatory features as inputs for speech synthesis and examines the outcomes after applying transfer learning from models based on more resourced languages. The project utilizes the ToucanTTS system from the University of Stuttgart, which is based on the FastSpeech2 architecture and converts text into articulatory features for speech synthesis. A TTS system for Quenya was developed by fine-tuning three models using a 34-minute Quenya dataset: one for Finnish, one for English, and a multilingual model provided by ToucanTTS. The results showed that the Finnish fine-tuned model produced better speech quality than the English fine-tuned model, while the multilingual fine-tuned model produced the most natural and accurate speech.


# Acknowledgements
I would like to express my deepest gratitude to my supervisor, Ph.D. Candidate Phat Do, and Associate Professor Dr. Matt Coler, for their invaluable guidance throughout this project. My thanks also go to the original creators of the [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan) system for providing such a convenient tool. Special thanks to the author of the book [Atanquesta](https://middangeard.org.uk/aglardh/atanquesta), whose open-source recordings were crucial to this project. I am also deeply grateful to [Glaemscrafu](https://glaemscrafu.jrrvf.com/english/index.html), a website rich with resources, particularly the recordings by Benjamin Babut and Bertrand Bellet. Additionally, I extend my gratitude to all the participants in the questionnaire survey and the Quenya enthusiasts who were willing to write detailed listening reports. Finally, I am deeply grateful to the volunteer speaker, Igor Marchenko. Despite your busy schedule leading up to graduation, you provided exceptionally high-quality recordings. You are the best classmate I have ever had.

# Experiment
Based on the tutorial from the original repository, I have added Quneya language (`qya`) support in the `Preprocessing\TextFrontend.py` file and created the Quneya language G2P script, which is also located in the Preprocessing directory.

…… update in the future……
