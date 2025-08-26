# Stimmenfaenger
conda activate Stimmenfaenger
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

python simple_test.py


if __name__ == '__main__':

    import os
    import sys
    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()

    from RealtimeSTT import AudioToTextRecorder

    recorder = AudioToTextRecorder(
        spinner=False,
        silero_sensitivity=0.01,
        model="tiny",
        language="de",
        )

    print("Say something...")
    
    try:
        while (True):
            print(recorder.text())
    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")




pip install transformers torch
pip install matplotlib
python bert3.py

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel
from RealtimeSTT import AudioToTextRecorder
import time

def analyse_attention(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    # ATTENTION: [layers, heads, seq_len, seq_len]
    attentions = torch.stack(outputs.attentions).squeeze(1)

    # 👉 Nur letzter Layer
    last_layer = attentions[-1]  # [heads, seq_len, seq_len]

    # 👉 CLS-Token Aufmerksamkeit auf alle Tokens
    cls_attention = last_layer[:, 0, :]  # [heads, seq_len]

    # 👉 Mittelwert über Heads
    token_importance = cls_attention.mean(dim=0)  # [seq_len]

    # 👉 Stärkere Ausschläge durch Potenz
    boosted = token_importance ** 2.5

    # 👉 Normierung (0–1)
    boosted = (boosted - boosted.min()) / (boosted.max() - boosted.min() + 1e-8)

    # Tokens zurückwandeln
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    print("\n📌 Token-Wichtigkeit (verstärkt durch Potenz + Normierung):")
    for token, score in zip(tokens, boosted):
        print(f"{token:>12} : {score.item():.4f}")
    print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":

    # DLL-Fix für Windows + Python < 3.10
    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()

    # 1. Lade deutsches BERT-Modell
    print("Lade BERT-Modell...")
    model_name = 'bert-base-german-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    print("Modell geladen.\n")

    # 2. Initialisiere STT
    recorder = AudioToTextRecorder(
        spinner=False,
        silero_sensitivity=0.01,
        model="tiny",
        language="de"
    )

    print("🎙️  Sprich etwas... (Strg+C zum Beenden)\n")

    try:
        while True:
            text = recorder.text()
            if text.strip():
                print(f"\n🗣️  Erkannt: {text}")
                analyse_attention(text, tokenizer, model)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n🚪 Beendet durch Benutzer.")


