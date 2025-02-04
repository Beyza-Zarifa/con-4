import numpy as np
import copy
from agents.agent_mtcs.mcts import generate_move_mcts
from game_utils import (
    apply_player_action, check_end_state, GameState, PLAYER1, PLAYER2, initialize_game_state, pretty_print_board
)


def simulate_game():
    """Simuliert ein komplettes Spiel mit MCTS und zeigt es in der Konsole."""
    board = initialize_game_state()
    player = PLAYER1
    game_data = []

    while True:
        print(pretty_print_board(board))
        print(f"Spieler {'X' if player == PLAYER1 else 'O'} ist am Zug...")

        move_data = generate_move_mcts(board, player, None, iterations=500)
        # Da generate_move_mcts immer (move, None) liefert, extrahieren wir den move:
        move = move_data[0] if isinstance(move_data, tuple) else move_data

        apply_player_action(board, move, player)
        game_data.append((copy.deepcopy(board), player))

        game_result = check_end_state(board, player)
        if game_result == GameState.IS_WIN:
            print(pretty_print_board(board))
            print(f"Spieler {'X' if player == PLAYER1 else 'O'} hat gewonnen!")
            return game_data, 1 if player == PLAYER1 else -1
        elif game_result == GameState.IS_DRAW:
            print(pretty_print_board(board))
            print("Das Spiel endet unentschieden!")
            return game_data, 0

        player = PLAYER1 if player == PLAYER2 else PLAYER2


def generate_dataset(num_games=1000, filename="game_data.npz"):
    """
    Generates multiple game simulations and saves the resulting data.

    Args:
        num_games (int): The number of games to simulate.
        filename (str): The file path where the dataset will be saved.
    """
    all_boards = []  # Stores all board states from all games
    all_results = []  # Stores the outcome of each game

    for i in range(num_games):
        print(f"Starting game {i + 1}...")
        game_data, result = simulate_game()
        for board, _ in game_data:
            all_boards.append(board)
            all_results.append(result)

        # Save data periodically to prevent data loss
        if (i + 1) % 10 == 0:
            np.savez_compressed(filename, boards=np.array(all_boards), results=np.array(all_results))
            print(f"Intermediate save after {i + 1} games...")

    # Final save of the dataset
    np.savez_compressed(filename, boards=np.array(all_boards), results=np.array(all_results))
    print(f"Final dataset saved to {filename}")

if __name__ == "__main__":
    generate_dataset(num_games=1000)  # Run 10 test games
