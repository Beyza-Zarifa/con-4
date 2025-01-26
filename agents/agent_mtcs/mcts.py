from game_utils import GameState, valid_moves, PLAYER1, PLAYER2, check_move_status, BoardPiece, SavedState
import random
import math
import numpy as np


class Node:
    def __init__(self, board: np.ndarray, player=None, parent=None, move=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move

        self.N = 0  # Number of visits
        self.Q = 0  # Total value (reward)
        self.children = {}  # Dictionary to map moves to child nodes
        self.gameState = GameState.STILL_PLAYING

    def is_fully_expanded(self):
        """Checks if all possible moves have been expanded into child nodes."""
        if self.gameState != GameState.STILL_PLAYING:
            return True  # Terminal nodes are fully expanded

        valid_moves_set = set(valid_moves(self.board))
        expanded_moves_set = set(self.children.keys())
        return valid_moves_set == expanded_moves_set

    def best_child(self, exploration_weight=math.sqrt(2)):
        """
        Determines the best child node based on the UCT formula.
        """
        if not self.children:
            print(f"[DEBUG] No children available for node with move {self.move}.")
            raise ValueError("Attempting to select a child from a node with no children.")

        def uct_value(child):
            if child.N == 0:
                return float('inf')  # Prioritize unexplored nodes
            return (child.Q / child.N) + exploration_weight * math.sqrt(math.log(self.N) / child.N)

        return max(self.children.values(), key=uct_value)

    def select_child(self):
        """
        Selects the child with the highest visit count (N).
        Typically used after the MCTS search to determine the final move.
        """
        if not self.children:
            print(f"[DEBUG] No children available for node with move {self.move}.")
            raise ValueError("Attempting to select a child from a node with no children.")

        best_child = None
        highest_visits = -1

        for child in self.children.values():
            if child.N > highest_visits:
                highest_visits = child.N
                best_child = child

        return best_child

    def expand(self):
        """
        Expands the current node by adding one new child node for an unvisited move.
        """
        if self.gameState != GameState.STILL_PLAYING:
            raise ValueError("Cannot expand a terminal node.")

        # Get the set of valid moves and already expanded moves
        valid_moves_set = set(valid_moves(self.board))
        expanded_moves_set = set(self.children.keys())

        # Find unvisited moves
        unvisited_moves = list(valid_moves_set - expanded_moves_set)

        if not unvisited_moves:
            print(f"[DEBUG] No unvisited moves left for node with move {self.move}.")
            raise ValueError("No moves left to expand; node is already fully expanded.")

        # Randomly select one unvisited move to expand
        move = random.choice(unvisited_moves)

        # Simulate the new board state resulting from this move
        new_board = apply_move(self.board, move, self.player)

        # Create the new child node
        child_node = Node(
            board=new_board,
            player=switch_player(self.player),  # Switch to the opponent's turn
            parent=self,
            move=move,
        )

        # Add the child to the children dictionary
        self.children[move] = child_node

        return child_node

    def simulate(self):
        """
        Runs a random simulation to completion, returning the game result (1 for PLAYER1 win, -1 for PLAYER2 win, 0 for draw).
        """
        # Make a copy of the current board for simulation (avoid modifying original board)
        simulation_board = self.board.copy()

        current_player = self.player  # Start with the current player from the node

        while True:
            # Get valid moves for the current player
            valid_moves_for_simulation = valid_moves(simulation_board)

            # If there are no valid moves left, it's a draw
            if not valid_moves_for_simulation:
                return 0  # Draw

            # Select a random valid move
            move = random.choice(valid_moves_for_simulation)

            # Apply the move to the board
            simulation_board = apply_move(simulation_board, move, current_player)

            # Check if the current player has won the game
            if check_move_status(simulation_board, current_player) == GameState.IS_WIN:
                return current_player

                # Switch to the other player for the next turn
            current_player = switch_player(current_player)

    def backpropagate(self, result):
        """
        Backpropagates the result of the simulation to the root node.

        Args:
            result (int): The result of the simulation (1 for PLAYER1 win, -1 for PLAYER2 win, 0 for draw).
        """
        # Start at the current node and propagate the result upwards
        current_node = self

        while current_node is not None:
            # Update the visit count (N)
            current_node.N += 1

            # Update the total value (Q) based on the simulation result
            if result == 1:  # PLAYER1 win
                current_node.Q += 1
            elif result == -1:  # PLAYER2 win
                current_node.Q -= 1
            # If it's a draw (a result == 0), no need to change Q

            # Move to the parent node for backpropagation
            current_node = current_node.parent


def apply_move(board, move, player):
    """
    Simulates applying a move to the board for the given player.

    Args:
        board (list of int): The current state of the board.
        move (int): The column where the player wants to place their token.
        player (int): The player making the move (1 or -1).

    Returns:
        list of list of int: A new board state after the move is applied.
    """
    # Create a copy of the board to avoid modifying the original
    new_board = [row[:] for row in board]

    # Find the lowest available row in the specified column
    for row in reversed(new_board):
        if row[move] == 0:
            row[move] = player
            break

    return new_board


def switch_player(current_player):
    if current_player == PLAYER1:
        return PLAYER2
    else:
        return PLAYER1


def mcts(board: np.ndarray, player: BoardPiece, iterations=1000):
    root_node = Node(board=board, player=player)

    for _ in range(iterations):
        node = root_node
        while not node.is_fully_expanded():
            node = node.select_child()  # Choose the best child

        if node.is_fully_expanded():
            node = node.expand()  # Expand a new child node

        result = node.simulate()  # Run a random simulation

        node.backpropagate(result)  # Backpropagate the result to update the tree

    # After the simulations, choose the best child (move) based on visit counts
    best_child = root_node.best_child(exploration_weight=0)
    return best_child.move


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: SavedState | None):
    action = mcts(board, player)
    return action, saved_state