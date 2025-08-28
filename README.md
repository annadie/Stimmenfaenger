# 🎙️ Stimmenfänger – Real-Time German Speech-to-Text with BERT Attention Analysis

This project combines real-time speech-to-text transcription with keyword extraction and attention scoring. Using a lightweight speech recognition model, it transcribes spoken German audio, then leverages KeyBERT to identify and score the most relevant keywords from the transcribed text. This tool can be useful for applications like voice-driven search, meeting transcription, or content analysis.

---

## 📦 Installation

Make sure you have Python 3.8–3.10 installed.

### 1. Set up the environment

```bash
conda create -n Stimmenfaenger python=3.10
conda activate Stimmenfaenger


pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install Keybert
pip install sentence-transformers
pip install RealtimeSTT
pip install PyTorch

--
annotated-types==0.7.0
av==15.0.0
beautifulsoup4==4.13.5
blis==1.3.0
captum==0.8.0
catalogue==2.0.10
certifi==2025.8.3
cffi==1.17.1
charset-normalizer==3.4.3
click==8.2.1
cloudpathlib==0.21.1
colorama==0.4.6
coloredlogs==15.0.1
confection==0.1.5
contourpy==1.3.2
ctranslate2==4.6.0
cycler==0.12.1
cymem==2.0.11
de_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl#sha256=fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7
docopt==0.6.2
emoji==2.14.1
enum34==1.1.10
faster-whisper==1.1.1
filelock==3.13.1
flatbuffers==25.2.10
fonttools==4.59.1
fsspec==2024.6.1
fuzzywuzzy==0.18.0
halo==0.0.31
huggingface-hub==0.34.4
humanfriendly==10.0
idna==3.10
jellyfish==1.2.0
Jinja2==3.1.4
joblib==1.5.1
Js2Py==0.74
keybert==0.9.0
kiwisolver==1.4.9
langcodes==3.5.0
language_data==1.3.0
Levenshtein==0.27.1
llvmlite==0.44.0
log-symbols==0.0.14
marisa-trie==1.3.1
markdown-it-py==4.0.0
MarkupSafe==2.1.5
matplotlib==3.10.5
mdurl==0.1.2
more-itertools==10.7.0
mpmath==1.3.0
murmurhash==1.0.13
networkx==3.3
nltk==3.9.1
numba==0.61.2
numpy==2.2.6
onnxruntime==1.22.1
openai-whisper==20250625
openwakeword==0.6.0
packaging==25.0
pandas==2.3.2
pillow==11.3.0
pipwin==0.5.2
preshed==3.0.10
protobuf==6.32.0
pvporcupine==1.9.5
PyAudio==0.2.14
pycparser==2.22
pydantic==2.11.7
pydantic_core==2.33.2
Pygments==2.19.2
pyjsparser==2.7.1
pyparsing==3.2.3
PyPrind==2.11.3
pyreadline3==3.5.4
pySmartDL==1.3.4
python-dateutil==2.9.0.post0
python-Levenshtein==0.27.1
pytz==2025.2
PyYAML==6.0.2
rake-nltk==1.0.6
RapidFuzz==3.13.0
realtimestt==0.3.104
regex==2025.7.34
requests==2.32.5
rich==14.1.0
safetensors==0.6.2
scikit-learn==1.7.1
scipy==1.15.2
seaborn==0.13.2
segtok==1.5.11
sentence-transformers==5.1.0
shellingham==1.5.4
six==1.17.0
smart_open==7.3.0.post1
sounddevice==0.5.2
soundfile==0.13.1
soupsieve==2.7
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
SpeechRecognition==3.14.3
spinners==0.0.24
srsly==2.5.1
stanza==1.10.1
sympy==1.13.3
tabulate==0.9.0
termcolor==3.1.0
thinc==8.3.6
threadpoolctl==3.6.0
tiktoken==0.11.0
tokenizers==0.21.4
tomli==2.2.1
torch==2.8.0+cpu
torchaudio==2.8.0+cpu
tqdm==4.67.1
transformers==4.55.4
typer==0.16.1
typing-inspection==0.4.1
typing_extensions==4.12.2
tzdata==2025.2
tzlocal==5.3.1
urllib3==2.5.0
wasabi==1.1.3
weasel==0.4.1
webrtcvad-wheels==2.0.14
websocket-client==1.8.0
websockets==15.0.1
wrapt==1.17.3
yake==0.6.0
--


# Speech-to-Text with Keyword Attention Scoring

This project combines **real-time speech-to-text transcription** with **keyword extraction and attention scoring** using [KeyBERT](https://github.com/MaartenGr/KeyBERT). The system listens to audio input, transcribes it in real time, and highlights the most relevant keywords based on semantic similarity.

---

## 🔍 Overview

- **Speech Recognition**: Real-time transcription using the [`RealtimeSTT`](https://github.com/Uberi/speech_recognition) interface.
- **Keyword Extraction**: Uses [KeyBERT](https://github.com/MaartenGr/KeyBERT) with the `sentence-transformers/LaBSE` model to score and extract keywords from the transcribed speech.
- **Language Support**: Configured for **German (de)** but can be adjusted.
- **Live Loop**: Continuously listens, transcribes, extracts, and prints keywords until interrupted.

---



