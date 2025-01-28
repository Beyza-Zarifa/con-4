import numpy as np
from collections import defaultdict
from game_utils import valid_moves, apply_player_action, check_end_state, GameState

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) tree.

    Attributes:
        board (np.ndarray): The current game board state.
        parent (MCTSNode): The parent node of this node.
        parent_action (tuple): The action that led to this node.
        children (list): A list of child nodes.
        num_visits (int): The number of times this node has been visited.
        results (defaultdict): A mapping of results (1 for win, -1 for loss, 0 for draw) to counts.
        untried_actions (list): The list of actions yet to be tried from this node.
    """

    def __init__(self, board, parent=None, parent_action=None):
        self.board = board
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.num_visits = 0
        self.results = defaultdict(int)
        self.untried_actions = valid_moves(board)

    def q_value(self):
        """Calculate the net wins for this node."""
        return self.results[1] - self.results[-1]

    def visit_count(self):
        """Get the number of visits to this node."""
        return self.num_visits

    def expand(self, player):
        """Expand a child node using one of the untried actions."""
        action = self.untried_actions.pop()
        new_board = self.board.copy()
        apply_player_action(new_board, action, player)
        child_node = MCTSNode(new_board, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        """Check if this node represents a terminal state."""
        return (
            check_end_state(self.board, 1) != GameState.STILL_PLAYING or
            check_end_state(self.board, 2) != GameState.STILL_PLAYING
        )

    def rollout(self, player):
        """
        Simulate a random playout from this node to a terminal state.

        Args:
            player (int): The current player (1 or 2).

        Returns:
            int: 1 for a win, -1 for a loss, 0 for a draw.
        """
        current_board = self.board.copy()
        current_player = player

        while (
            check_end_state(current_board, 1) == GameState.STILL_PLAYING and
            check_end_state(current_board, 2) == GameState.STILL_PLAYING
        ):
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
        """
        Update this node and propagate the result back to the root.

        Args:
            result (int): The outcome of the simulation (1 for win, -1 for loss, 0 for draw).
        """
        self.num_visits += 1
        self.results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        """Check if all possible actions have been tried."""
        return len(self.untried_actions) == 0

    def best_child(self, exploration_param=0.1):
        """
        Select the child node with the best UCT (Upper Confidence Bound for Trees) value.

        Args:
            exploration_param (float): The exploration parameter (default is 0.1).

        Returns:
            MCTSNode: The child node with the best UCT value.
        """
        choices_weights = [
            (child.q_value() / (child.visit_count() + 1e-6)) +
            exploration_param * np.sqrt(
                (2 * np.log(self.visit_count() + 1) / (child.visit_count() + 1e-6))
            )
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def tree_policy(self, player):
        """
        Navigate the tree, expanding if necessary, until a terminal node is reached.

        Args:
            player (int): The current player (1 or 2).

        Returns:
            MCTSNode: The terminal node reached.
        """
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand(player)
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, player, simulations=100):
        """
        Perform MCTS to find the best action from this node.

        Args:
            player (int): The current player (1 or 2).
            simulations (int): The number of simulations to run (default is 100).

        Returns:
            tuple: The action leading to the best outcome.
        """
        for _ in range(simulations):
            node_to_explore = self.tree_policy(player)
            reward = node_to_explore.rollout(player)
            node_to_explore.backpropagate(reward)

        return self.best_child(exploration_param=0).parent_action

def generate_move_mcts(board, player, saved_state):
    """
    Generate the best move using Monte Carlo Tree Search (MCTS).

    Args:
        board (np.ndarray): The current game board state.
        player (int): The current player (1 or 2).
        saved_state: A saved state object (unused in this implementation).

    Returns:
        tuple: The best action and the saved state (unchanged).
    """
    root = MCTSNode(board)
    best_action = root.best_action(player)
    return best_action, saved_state