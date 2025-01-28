import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from game_utils import valid_moves, apply_player_action, check_end_state, GameState

class Connect4Net(nn.Module):
    """
    Neural network for Connect Four.
    The network takes the game board as input and outputs action probabilities for each column.
    """
    def __init__(self):
        super(Connect4Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1),  # Convolutional layer to extract spatial features
            nn.ReLU(),  # Activation function
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # Second convolutional layer
            nn.ReLU()  # Activation function
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 2 * 4, 256),  # Fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(256, 7),  # Output layer for 7 columns
        )

    def forward(self, x):
        """
        Forward pass of the network.
        :param x: Input tensor representing the game board.
        :return: Output tensor with probabilities for each column.
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the output of convolutional layers
        return self.fc(x)

class NeuralAgent:
    """
    Neural agent for Connect Four.
    This class handles the decision-making, training, and memory for the neural network.
    """
    def __init__(self):
        """
        Initializes the neural agent with a neural network, optimizer, and memory storage.
        """
        self.model = Connect4Net()  # Neural network model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Optimizer
        self.loss_fn = nn.CrossEntropyLoss()  # Loss function for training
        self.memory = []  # Memory to store (state, action, reward) for training

    def choose_action(self, board, valid_actions):
        """
        Selects an action based on the current board state and valid actions.
        :param board: The current game board as a NumPy array.
        :param valid_actions: List of valid columns where a move can be made.
        :return: The selected action (column index).
        """
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            action_probs = self.model(board_tensor).squeeze()  # Predict action probabilities
        action_probs = action_probs.numpy()
        # Mask invalid actions
        masked_probs = np.full(action_probs.shape, -np.inf)
        masked_probs[valid_actions] = action_probs[valid_actions]  # Assign probabilities only to valid actions
        return np.argmax(masked_probs)  # Select the action with the highest probability

    def store_transition(self, state, action, reward):
        """
        Stores a transition (state, action, reward) for training.
        :param state: The game board state as a NumPy array.
        :param action: The action taken (column index).
        :param reward: The reward received for the action.
        """
        self.memory.append((state, action, reward))

    def train(self):
        """
        Trains the neural network using the stored transitions in memory.
        """
        if not self.memory:
            return

        # Prepare training data
        states, actions, rewards = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(1)  # Add channel dimension
        actions = torch.tensor(actions, dtype=torch.long)  # Action indices
        rewards = torch.tensor(rewards, dtype=torch.float32)  # Rewards

        self.model.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Clear previous gradients

        predictions = self.model(states)  # Predict action probabilities
        loss = self.loss_fn(predictions, actions)  # Compute loss
        loss.backward()  # Backpropagate the error
        self.optimizer.step()  # Update the model parameters

        # Clear memory after training
        self.memory = []

def generate_move_neural(board, player, saved_state):
    """
    Generates a move for the neural agent.
    :param board: The current game board as a NumPy array.
    :param player: The current player (1 or 2).
    :param saved_state: The saved state of the agent (NeuralAgent instance).
    :return: A tuple containing the selected action and the updated saved state.
    """
    if saved_state is None:
        saved_state = NeuralAgent()  # Initialize the agent if no state is saved

    valid_actions = valid_moves(board)  # Get the list of valid moves
    action = saved_state.choose_action(board, valid_actions)  # Select an action using the agent

    # Store state, action, reward (reward is 0 now, updated post-game)
    saved_state.store_transition(board.copy(), action, 0)

    return action, saved_state