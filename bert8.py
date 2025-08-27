import speech_recognition as sr
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import numpy as np

# Beispiel-Werte für bekannte Tokens (case-sensitive)
word_values = {
    "Die": 0.10,
    "kluge": 0.15,
    "Eule": 0.30,
    "fliegt": 0.20,
    "lautlos": 0.12,
    "durch": 0.05,
    "die": 0.03,
    "Nacht": 0.25,
}

tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")

def transcribe_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🎤 Sprich jetzt...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("🔎 Erkenne Sprache...")
        text = recognizer.recognize_google(audio, language="de-DE")
        print(f"\n➡️ Erkannt: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Sprache konnte nicht erkannt werden.")
        return None
    except sr.RequestError as e:
        print(f"❌ API-Fehler: {e}")
        return None

def scale_values(values, new_min=0.01, new_max=0.1):
    # Werte auf neuen Bereich skalieren
    arr = np.array(values)
    if np.ptp(arr) == 0:  # alle Werte gleich
        return [new_min] * len(values)
    scaled = (arr - arr.min()) / np.ptp(arr)  # Normierung 0-1
    scaled = scaled * (new_max - new_min) + new_min
    return scaled.tolist()

def assign_values_to_tokens(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Subtoken-Zusammenführung
    merged_tokens = []
    current_word = ""
    for token in tokens:
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                merged_tokens.append(current_word)
            current_word = token
    if current_word:
        merged_tokens.append(current_word)

    values = []
    unknown_tokens = []

    # Erstmal Werte oder None für jedes Token sammeln
    for token in merged_tokens:
        if token in word_values:
            values.append(word_values[token])
        else:
            values.append(None)
            unknown_tokens.append(token)

    # Für unbekannte Tokens Wert berechnen (basierend auf Wortlänge)
    unknown_values_raw = [len(tok) for tok in unknown_tokens]
    scaled_unknown_values = scale_values(unknown_values_raw, new_min=0.01, new_max=0.1)

    # Jetzt None durch skalierte Werte ersetzen
    idx = 0
    for i, val in enumerate(values):
        if val is None:
            values[i] = scaled_unknown_values[idx]
            idx += 1

    return merged_tokens, values

def plot_tokens(tokens, values):
    plt.figure(figsize=(10, 4))
    plt.bar(tokens, values, color="skyblue")
    plt.title("Token Werte (Beispielwerte + skaliert für unbekannte Tokens)")
    plt.xlabel("Token")
    plt.ylabel("Wert")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    while True:
        text = transcribe_speech()
        if text:
            tokens, values = assign_values_to_tokens(text)
            print("\nToken und zugehörige Werte:\n")
            for t, v in zip(tokens, values):
                print(f"{t:>10} : {v:.3f}")
            plot_tokens(tokens, values)
        else:
            print("Bitte erneut sprechen...\n")
