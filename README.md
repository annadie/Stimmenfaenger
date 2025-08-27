# 🎙️ Stimmenfänger – Real-Time German Speech-to-Text with BERT Attention Analysis

**Stimmenfänger** is a real-time speech-to-text system using [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection and Whisper (via `RealTimeSTT`) for transcription. It also includes a BERT-based attention analysis module to highlight which words or tokens are most important in the recognized sentence.

---

## 📦 Installation

Make sure you have Python 3.8–3.10 installed.

### 1. Set up the environment

```bash
conda create -n Stimmenfaenger python=3.10
conda activate Stimmenfaenger


pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install transformers matplotlib

cd C:\Users\simon\Documents\Heidenheim\Stimmenfaenger\RealTimeSTT-\tests
python simple_test.py
python bert3.py
