from model import train_model, Connect4NN
from data_handler import save_training_data, load_training_data, save_model, load_model
from agents.agent_MCTS.MCTS import simulate
import numpy as np
from game_utils import NO_PLAYER, PLAYER1, PLAYER2

def generate_training_data(num_samples=1000):
    training_data = []
    for _ in range(num_samples):
        board = np.random.choice([NO_PLAYER, PLAYER1, PLAYER2], size=(6, 7))
        score = simulate(board, PLAYER1)  # Statt evaluate_board
        training_data.append((board, score))
    return training_data


if __name__ == "__main__":
    model = Connect4NN()
    load_model(model)
    training_data = load_training_data()
    
    if not training_data:
        training_data = generate_training_data(1000)
        save_training_data(training_data)

    train_model(training_data, model, epochs=100)
    save_model(model)
    

