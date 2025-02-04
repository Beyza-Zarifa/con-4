import json
import torch
import numpy as np

MODEL_FILE = "connect4_model.pth"
TRAINING_DATA_FILE = "training_data.json"

def save_training_data(training_data):
    with open(TRAINING_DATA_FILE, "w") as f:
        json.dump(
            [{"board": board.tolist(), "score": score} for board, score in training_data], f
        )

def load_training_data():
    try:
        with open(TRAINING_DATA_FILE, "r") as f:
            data = f.read().strip()  # Entfernt Leerzeichen und Zeilenumbrüche
            if not data:
                return []  # Falls Datei leer ist, gib eine leere Liste zurück
            json_data = json.loads(data)
            return [(np.array(entry["board"]), entry["score"]) for entry in json_data]
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # Falls Datei fehlt oder fehlerhaft ist, gib eine leere Liste zurück

def save_model(model):
    torch.save(model.state_dict(), MODEL_FILE)

def load_model(model):
    try:
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
        print("Modell erfolgreich geladen.")
    except FileNotFoundError:
        print("Kein Modell gefunden, starte neu.")
