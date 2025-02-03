import copy
import numpy as np
import math
from game_utils import valid_moves, apply_player_action, check_end_state, GameState, PLAYER1, PLAYER2


class MCTSNode:
    def __init__(self, board, parent=None, move=None, player=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.move = move
        self.player = player
        self.untried_moves = valid_moves(board)

    def uct_value(self, exploration_param=1.4):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_param * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, exploration_param=1.4):
        return max(self.children, key=lambda child: child.uct_value(exploration_param))

    def expand(self, player):
        if not self.untried_moves:
            return None
        move = self.untried_moves.pop()
        new_board = copy.deepcopy(self.board)
        apply_player_action(new_board, move, player)
        child_node = MCTSNode(new_board, parent=self, move=move, player=player)
        self.children.append(child_node)
        return child_node

    def simulate(self):
        current_board = copy.deepcopy(self.board)
        current_player = self.player

        while True:
            moves = valid_moves(current_board)
            if not moves:
                return GameState.IS_DRAW

            move = np.random.choice(moves)
            apply_player_action(current_board, move, current_player)
            game_result = check_end_state(current_board, current_player)

            if game_result == GameState.IS_WIN:
                return 1 if current_player == self.player else -1
            elif game_result == GameState.IS_DRAW:
                return 0

            current_player = PLAYER1 if current_player == PLAYER2 else PLAYER2

    def backpropagate(self, result):
        current_node = self
        while current_node is not None:
            current_node.visits += 1
            current_node.wins += result
            current_node = current_node.parent


def generate_move_mcts(board, player, save_state, iterations=1000):
    root = MCTSNode(board, player=player)

    for _ in range(iterations):
        node = root
        while node.children and not node.untried_moves:
            node = node.best_child()

        if node.untried_moves:
            node = node.expand(player)

        if node:
            result = node.simulate()
            node.backpropagate(result)

    return root.best_child(0).move, save_state
