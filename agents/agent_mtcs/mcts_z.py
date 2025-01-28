import numpy as np
from collections import defaultdict
from game_utils import valid_moves, apply_player_action, check_end_state, GameState

class Node:
    def __init__(self, board, parent=None, parent_action=None):
        self.board = board
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._untried_actions = valid_moves(board)

    def q(self):
        # Net wins for this node.
        return self._results[1] - self._results[-1]

    def n(self):
        # Number of visits to this node.
        return self._number_of_visits

    def expand(self, player):
        # Expand a child node using one of the untried actions.
        action = self._untried_actions.pop()
        new_board = self.board.copy()
        apply_player_action(new_board, action, player)
        child_node = Node(new_board, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        # Check if this node represents a terminal state.
        return check_end_state(self.board, 1) != GameState.STILL_PLAYING or check_end_state(self.board, 2) != GameState.STILL_PLAYING

    def rollout(self, player):
        # Simulate a random playout from this node to a terminal state.
        current_board = self.board.copy()
        current_player = player

        while check_end_state(current_board, 1) == GameState.STILL_PLAYING and check_end_state(current_board, 2) == GameState.STILL_PLAYING:
            possible_moves = valid_moves(current_board)
            action = np.random.choice(possible_moves)
            apply_player_action(current_board, action, current_player)
            current_player = 3 - current_player  # Switch player (1 -> 2, 2 -> 1).

        if check_end_state(current_board, player) == GameState.IS_WIN:
            return 1
        elif check_end_state(current_board, 3 - player) == GameState.IS_WIN:
            return -1
        else:
            return 0

    def backpropagate(self, result):
        # Update this node and propagate the result back to the root.
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        # Check if all possible actions have been tried.
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        # Select the child with the best UCT value.
        choices_weights = [
            (child.q() / (child.n() + 1e-6)) + c_param * np.sqrt((2 * np.log(self.n() + 1) / (child.n() + 1e-6)))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def tree_policy(self, player):
        # Navigate the tree, expanding if necessary, until a terminal node is reached.
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand(player)
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, player, simulations=100):
        # Perform MCTS to find the best action from this node.
        for _ in range(simulations):
            v = self.tree_policy(player)
            reward = v.rollout(player)
            v.backpropagate(reward)

        return self.best_child(c_param=0).parent_action

def generate_move_mcts(board, player, saved_state):
    root = Node(board)
    best_action = root.best_action(player)
    return best_action, saved_state