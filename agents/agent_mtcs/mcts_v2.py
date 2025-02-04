import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import defaultdict

from game_utils import valid_moves, apply_player_action


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Store the action that led to this node
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = valid_moves(state)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.0):
        choices_weights = [
            (child.value / (child.visits + 1e-6))
            + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, player):
        if not self.untried_actions:
            return None  # No more moves to expand

        action = self.untried_actions.pop()
        new_state = self.state.copy()
        apply_player_action(new_state, action, player)
        child_node = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward


def simulate_random_playout(state):
    return random.choice([-1, 1])


def generate_move_mcts(state, player, saved_state=None, num_simulations=50000):
    root = MCTSNode(state)
    for _ in range(num_simulations):
        node = root
        while not node.untried_actions and node.children:
            node = node.best_child()
        if node.untried_actions:
            node = node.expand(player)
        reward = simulate_random_playout(node.state)
        while node is not None:
            node.update(reward)
            node = node.parent

    best_move_node = root.best_child(exploration_weight=0)
    return best_move_node.action, saved_state
