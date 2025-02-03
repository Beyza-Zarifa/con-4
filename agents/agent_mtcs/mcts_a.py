"""
Monte Carlo Tree Search (MCTS) algorithm.
A decision-making algorithm that selects the best move in a game by simulating multiple random playthroughs.
"""

import copy
import random
from math import sqrt, log
from game_utils import valid_moves, apply_player_action, check_end_state, GameState


class Node:
    """
    A node in the Monte Carlo Tree Search (MCTS) tree.
    Contains a game state, a reference to the parent node, a list of child nodes, and statistics for the node.

    Parameters:
        state: The game state represented by the node.
        parent: The parent node of the current node. Default is None.

    Attributes:
        state: The game state represented by the node.
        parent: The parent node of the current node.
        children: A list of child nodes of the current node.
        visits: The number of times the node has been visited.
        wins: The number of wins from the node.

    Methods:
        is_fully_expanded() -> bool: Returns True if all valid moves from this state have been expanded as child nodes.
        best_child(exploration_weight: float = 1.4) -> "Node": Selects the child node with the best balance of exploration and exploitation, using the UCT/UCB1 formula.
    """
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self) -> bool:
        """
        Returns True if all valid moves from this state have been expanded as child nodes.
        """
        return len(self.children) == len(self.state.valid_moves())

    def best_child(self, exploration_weight: float = 1.4) -> "Node": # TODO-?: Why "Node" and not just Node?
        """
        Selects the child node with the best balance of exploration and exploitation,
        using the UCT/UCB1 formula.
        """
        return max(
            self.children,
            key=lambda child: (child.wins / child.visits) +
                              exploration_weight * sqrt(log(self.visits) / child.visits)
        )


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) algorithm.
    A decision-making algorithm that selects the best move in a game by simulating multiple random playthroughs.

    Parameters:
        game_constructor: A function that returns a new game state.
        player_1: The player that starts the game.
        player_2: The player that follows.
        iterations: The number of iterations to run the MCTS algorithm. Default is 1000.

    Attributes:
        game_constructor: A function that returns a new game state.
        player_1: The player that starts the game.
        player_2: The player that follows.
        iterations: The number of iterations to run the MCTS algorithm.

    Methods:
        search(root: Node) -> Node: Repeats the MCTS process (selection, expansion, simulation, backpropagation)
                                    for a given number of iterations. Returns the best child node after the iterations.
        get_changed_position(list1, list2) -> tuple[int, int]: Returns the changed position of the bord.
        _select(node: Node) -> Node: Traverses the tree from the root, choosing the best child (based on UCT, UCB1),
                                     until reaching an unexpanded node or a terminal state.
        _expand(node: Node) -> Node: Expands a node by generating a new child node for an unvisited move.
                                     Creates a new game state for the selected move.
        _simulate(state: Node) -> int: Simulates a random playthrough from the current state until the game ends.
                                       Assigns a reward: +1 if ??? wins, -1 if ??? wins, 0 for a draw.
        _backpropagate(node: Node, reward: int) -> None: Updates the wins and visits of the node and its ancestors,
                                                         alternating the reward (+1 for one player, -1 for the other).
        _get_next_state(state: Node, move: int) -> Node: Creates a copy of the current state and
                                                         applies a move to generate a new state.
        _clone_state(state: Node) -> Node: Creates a deep copy of the game state
                                           to ensure modifications do not affect the original.
    """

    def __init__(self, game_constructor, player_1: str, player_2: str, iterations: int = 1000) -> None:
        self.game_constructor = game_constructor
        self.player_1 = player_1
        self.player_2 = player_2
        self.iterations = iterations

    def search(self, root: Node) -> Node:
        """
        Repeats the MCTS process (selection, expansion, simulation, backpropagation) for a given number of iterations.

        Arguments:
            root: The root node of the MCTS tree.

        Returns:
            Node: The best child node after the iterations.
        """
        for _ in range(self.iterations):
            node = self._select(node=root)
            # TODO-print:
            # node.state.display_board()
            # print(f"{node.state.player = }")
            reward = self._simulate(state=node.state)
            # TODO-print:
            # print(f"{reward = }")
            self._backpropagate(node=node, reward=reward)
        # TODO-print:
        # for node in root.children:
        #     node.state.display_board()
        #     print(f"{node.wins = }")
        #     print(f"{node.visits = }")
        return root.best_child(exploration_weight=0.0)

    def get_changed_position(self, list1, list2) -> tuple[int, int]:
        """
        Returns the changed position of the bord. (the position of the first element that differs between two lists)

        Arguments:
            list1: The first list to compare.
            list2: The second list to compare.

        Returns:
            tuple[int, int]: The position of the first element that differs between the two lists.
        """
        pos_y = [index for index, (element1, element2) in enumerate(zip(list1, list2)) if element1 != element2][0]
        pos_x = [index for index, (element1, element2) in enumerate(zip(list1[pos_y], list2[pos_y])) if element1 != element2][0]
        return (pos_y, pos_x)

    def _select(self, node: Node) -> Node:
        """
        Traverses the tree from the root, choosing the best child (based on UCT, UCB1),
        until reaching an unexpanded node or a terminal state.

        Arguments:
            node: The current node in the tree.

        Returns:
            Node: The selected node.
        """
        while not node.state.is_game_over():
            # TODO-print:
            # n_empty = len([x for row in node.state.board for x in row if x == " "])
            # if n_empty < 40:
            #     print(f"{n_empty = }")
            if not node.is_fully_expanded():
                return self._expand(node=node)
            node = node.best_child()
        return node

    def _expand(self, node: Node) -> Node:
        """
        Expands a node by generating a new child node for an unvisited move.
        Creates a new game state for the selected move.

        Arguments:
            node: The current node to expand.

        Returns:
            Node: The new child node.
        """
        moves = node.state.get_valid_moves()
        for move in moves:
            state_new = self._get_next_state(state=node.state, move=move)
            if not any(child.state.board == state_new.board for child in node.children):
                node_child = Node(state=state_new, parent=node)
                node.children.append(node_child)
                return node_child
        raise Exception("All moves have been visited.")

    def _simulate(self, state: Node) -> int:
        """
        Simulates a random playthrough from the current state until the game ends.

        Arguments:
            state: The current game state.

        Returns:
            int: assigned rewards:
                 +1 if player_rewarded wins,
                 -1 if player_rewarded loses,
                 0 for a draw.
        """
        state_cloned = self._clone_state(state=state)
        player_rewarded = self.player_1 if state_cloned.player == self.player_2 else self.player_2 # player wo made the last step to get current state

        # TODO-print:
        # print(f"{state_cloned.player = }")
        # print(f"{player_rewarded = }")
        while not state_cloned.is_game_over():
            move = random.choice(state_cloned.get_valid_moves())
            state_cloned.make_move(move=move)
        # TODO-print:
        # current_state.display_board()
        if state_cloned.is_winner(player=player_rewarded):
            return 1
        if state_cloned.is_winner(player=state_cloned.player):
            return -1
        return 0

    def _backpropagate(self, node: Node, reward: int) -> None:
        """
        Updates the wins and visits of the node and its ancestors,
        alternating the reward (+1 for one player, -1 for the other).

        Arguments:
            node: The current node.
            reward: The reward assigned to the node.
        """
        while node is not None:
            node.visits += 1
            node.wins += reward
            reward = -reward
            node = node.parent

    def _get_next_state(self, state: Node, move: int) -> Node:
        """
        Creates a copy of the current state and
        applies a move to generate a new state.
        Arguments:
            state: The current game state.
            move: The move to apply to the state.
        Returns:
            Node: The new game state after applying the move.
        """
        state_new = self._clone_state(state=state)
        state_new.make_move(move=move)
        return state_new

    def _clone_state(self, state: Node) -> Node:
        """
        Creates a deep copy of the game state
        to ensure modifications do not affect the original.
        Arguments:
            state: The game state to clone.
        Returns:
            Node: A deep copy of the game state.
        """
        state_new = self.game_constructor()
        state_new.board = copy.deepcopy(state.board)
        state_new.player = state.player
        return state_new


def generate_move_mcts(board, player, saved_state):
    """
    Generates the best move using the Monte Carlo Tree Search (MCTS) algorithm.

    Arguments:
        board: The current game state.
        player: The current player.
        saved_state: The saved state for potential reuse (not used in MCTS).

    Returns:
        tuple: (best_action, saved_state)
    """
    # Ensure the board is correctly cloned
    board_copy = board.copy()

    # Initialize MCTS with the game constructor and players
    mcts = MCTS(game_constructor=lambda: board_copy, player_1=player, player_2='O' if player == 'X' else 'X')

    # Create root node with the current board state
    root = Node(state=board_copy)

    # Perform MCTS search to find the best move
    best_node = mcts.search(root=root)

    # Determine the action (move) that led to the best node
    best_action = mcts.get_changed_position(board.board, best_node.state.board)

    return best_action, saved_state

