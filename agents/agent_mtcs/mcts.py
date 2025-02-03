import copy
import numpy as np
import math

from game_utils import valid_moves, apply_player_action, check_end_state, GameState, PLAYER1, PLAYER2


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search (MCTS) algorithm.
    Each node represents a game state and stores statistical information
    about its visits, wins, and possible moves.
    """

    def __init__(self, board, parent=None, move=None, player=None):
        """
        Initializes the MCTS node.

        :param board: The current game board state.
        :param parent: The parent node in the tree.
        :param move: The move that led to this node.
        :param player: The player who made the move.
        """
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.move = move
        self.player = player
        self.untried_moves = valid_moves(board)

    def uct_value(self, exploration_param=1.4):
        """
        Computes the Upper Confidence Bound for Trees (UCT) value.

        :param exploration_param: The exploration factor.
        :return: UCT value for node selection.
        """
        if self.visits == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return (self.wins / self.visits) + exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, exploration_param=1.4):
        """
        Selects the best child node based on the UCT value.

        :param exploration_param: The exploration factor.
        :return: The best child node.
        """
        return max(self.children, key=lambda child: child.uct_value(exploration_param))

    def expand(self, player):
        """
        Expands the tree by creating a new child node from an untried move.

        :param player: The player making the move.
        :return: The newly created child node.
        """
        if not self.untried_moves:
            return None  # No more moves to expand

        move = self.untried_moves.pop()
        new_board = copy.deepcopy(self.board)
        apply_player_action(new_board, move, player)
        child_node = MCTSNode(new_board, parent=self, move=move, player=player)
        self.children.append(child_node)
        return child_node

    def simulate(self):
        """
        Simulates a random game from the current state until a terminal state is reached.

        :return: +1 if the original player wins, -1 if they lose, 0 if they draw.
        """
        current_board = copy.deepcopy(self.board)
        current_player = self.player

        while True:
            moves = valid_moves(current_board)
            if not moves:
                return GameState.IS_DRAW

            move = np.random.choice(moves)  # Select a random move
            apply_player_action(current_board, move, current_player)
            game_result = check_end_state(current_board, current_player)

            if game_result == GameState.IS_WIN:
                return 1 if current_player == self.player else -1
            elif game_result == GameState.IS_DRAW:
                return 0

            # Switch players
            current_player = PLAYER1 if current_player == PLAYER2 else PLAYER2

    def backpropagation(self, result):
        """
        Updates the statistics of the nodes along the path back to the root.

        :param result: The result of the simulation (win/loss/draw).
        """
        current_node = self
        while current_node is not None:
            current_node.visits += 1
            current_node.wins += result
            current_node = current_node.parent


def forced_move(board, player):
    """
    Checks if there is a forced move (win or loss) for the current player.

    :param board: The current game board state.
    :param player: The player making the move.
    :return: The move that forces a win or loss or None if no such move exists.
    """
    valid_moves_list = valid_moves(board)

    # Check if any move results in a win for the current player
    for move in valid_moves_list:
        new_board = copy.deepcopy(board)
        apply_player_action(new_board, move, player)
        if check_end_state(new_board, player) == GameState.IS_WIN:
            return move  # Found a move that leads to a win

    # Check if any move results in a loss for the current player (opponent's win)
    opponent = PLAYER1 if player == PLAYER2 else PLAYER2
    for move in valid_moves_list:
        new_board = copy.deepcopy(board)
        apply_player_action(new_board, move, opponent)
        if check_end_state(new_board, opponent) == GameState.IS_WIN:
            return move  # Found a move that leads to a loss (avoid this move)

    # No forced win or loss move, return None
    return None


def generate_move_mcts(board, player, save_state, iterations=1):
    """
    Generates a move using the Monte Carlo Tree Search algorithm.

    :param board: The current game board state.
    :param player: The player making the move.
    :param save_state: Placeholder for external state tracking.
    :param iterations: The number of MCTS iterations to perform.
    :return: The best move is determined by MCTS and the save_state.
    """

    forced_move_result = forced_move(board, player)
    if forced_move_result is not None:
        return forced_move_result, save_state  # Return the forced move before MCT

    root = MCTSNode(board, player=player)

    for _ in range(iterations):
        node = root

        # Selection: Traverse the tree until a leaf node is found
        while node.children and not node.untried_moves:
            node = node.best_child()

        # Expansion: If there are untried moves, expand the node
        if node.untried_moves:
            node = node.expand(player)

        # Simulation: Play a random game from this node
        if node:
            result = node.simulate()
            node.backpropagation(result)

    # Return the best move found (without an exploration factor)
    return root.best_child().move, save_state
