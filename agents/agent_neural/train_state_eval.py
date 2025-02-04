import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Funktion zum Laden der Spieldaten
def load_data(filename="game_data.npz"):
    data = np.load(filename)
    boards = torch.tensor(data["boards"], dtype=torch.float32)  # Shape (N, 6, 7)
    results = torch.tensor(data["results"], dtype=torch.float32).view(-1, 1)  # Shape (N, 1)
    return boards, results

# Daten laden
boards, results = load_data()

# Erstelle DataLoader für das Training
train_dataset = TensorDataset(boards, results)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(f"Daten geladen: {len(boards)} Spielzustände")
