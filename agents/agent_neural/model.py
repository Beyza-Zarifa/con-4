import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Connect4NN(nn.Module):
    def __init__(self):
        super(Connect4NN, self).__init__()
        self.fc1 = nn.Linear(6 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def board_to_tensor(board: np.ndarray) -> torch.Tensor:
    return torch.tensor(board.flatten(), dtype=torch.float32)

nn_model = Connect4NN()
optimizer = optim.Adam(nn_model.parameters(), lr=0.005)
loss_fn = nn.BCELoss()

def evaluate_board(board: np.ndarray, model) -> float:
    tensor_board = board_to_tensor(board).unsqueeze(0)
    with torch.no_grad():
        return model(tensor_board).item()

def train_model(training_data, model, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for board, target_value in training_data:
            optimizer.zero_grad()
            input_tensor = board_to_tensor(board).unsqueeze(0)
            prediction = model(input_tensor)
            loss = loss_fn(prediction, torch.tensor([[target_value]], dtype=torch.float32))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_data)}")
