from enum import Enum
from typing import Callable, Optional, Any
import math

import numpy as np

BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (BOARD_ROWS, BOARD_COLS)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class MoveStatus(Enum):
    IS_VALID = 1
    WRONG_TYPE = 'Input is not a number.'
    NOT_INTEGER = ('Input is not an integer, or isn\'t equal to an integer in '
                   'value.')
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns a ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).

    :return: shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Converts the game board into a human-readable string representation for display purposes.

    The bottom-left corner of the string representation corresponds to `board[0, 0]` in the array.
    Each cell is represented by a player symbol ("X" for PLAYER1, "O" for PLAYER2) or empty spaces
    for unoccupied cells.

    Example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |

    :param board: The current state of the game board as a 2D NumPy array.
    :return: A string representation of the board in a visually appealing format.
    """

    # Flip the board vertically to align the bottom-left corner with `board[0, 0]`.
    flipped_board = np.flipud(board)

    # Get the number of rows and columns in the board.
    num_rows, num_cols = flipped_board.shape

    # Initialize an empty string to build the board representation.
    board_representation = ""

    # Add the top boundary of the board.
    board_representation += "|" + "=" * (num_cols * 2) + "|\n"

    # Define symbols for each board state (PLAYER1, PLAYER2, empty).
    symbols = {PLAYER1: "X ", PLAYER2: "O ", NO_PLAYER: "  "}

    # Iterate through each row in the flipped board and construct the row string.
    for row in flipped_board:
        board_representation += "|" + "".join(symbols.get(col, "  ") for col in row) + "|\n"

    # Add the bottom boundary of the board.
    board_representation += "|" + "=" * (num_cols * 2) + "|\n"

    # Add column numbers at the bottom for easy reference.
    board_representation += "|"
    for i in range(num_cols):
        board_representation += f"{i} "
    board_representation += "|\n"

    # Return the complete board string representation.
    return board_representation


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Converts a string representation of the board (from `pretty_print_board`)
    back into a NumPy array. Useful for debugging when the board state is
    saved as a string after a crash.

    :param pp_board: The string representation of the board.
    :return: The reconstructed game board as a 2D NumPy array.
    """

    # Clean the board string to remove unwanted characters and boundaries.
    pp_board = clear_unwanted_characters(pp_board)

    # Mapping characters to board piece values.
    char_to_piece = {
        'X': PLAYER1,
        'O': PLAYER2,
        ' ': NO_PLAYER
    }

    # Convert characters to numeric values and reshape into the board's dimensions.
    pieces = [char_to_piece[char] for char in pp_board if char in char_to_piece]
    new_board = np.array(pieces, dtype=BoardPiece).reshape(BOARD_ROWS, BOARD_COLS)

    # Flip the board vertically to ensure the bottom row in the string matches the bottom row in the array.
    return np.flipud(new_board)


def clear_unwanted_characters(board_representation: str) -> str:
    """
    Removes unnecessary characters (e.g., boundaries, numbering) from the string representation
    of the board, leaving only the board's pieces.

    :param board_representation: The string containing the board with extra characters.
    :return: A cleaned string with only the board's piece data.
    """

    # Remove '=' and '|' boundary symbols.
    board_representation = board_representation.replace("=", "").replace("|", "")

    # Trim off leading and trailing unnecessary characters.
    board_representation = board_representation[1:-1]

    # Remove column numbering from the string.
    for col in range(BOARD_COLS):
        board_representation = board_representation.replace(str(col) + " ", "")

    # Remove unnecessary spaces in the string.
    board_representation = "".join([x for i, x in enumerate(board_representation) if i % 2 != 0 or x != " "])

    return board_representation[:-1]  # Trim any trailing spaces or unnecessary characters.


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Applies a player's action to the board by placing their piece in the lowest
    available row of the chosen column. The board is modified in place.

    :param board: The game board as a 2D NumPy array.
    :param action: The column index where the player wants to place their piece.
    :param player: The player's identifier (e.g., PLAYER1 or PLAYER2).
    :return:
    """
    for row in range(INDEX_LOWEST_ROW, BOARD_ROWS):
        if board[row, action] == NO_PLAYER:
            board[row, action] = player  # Place the piece in the first available row.
            break  # Exit the loop after placing the piece.


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Checks if there are four adjacent pieces equal to `player` arranged in a horizontal,
    vertical, or diagonal line.

    :param board: The game board as a 2D NumPy array.
    :param player: The player's identifier (e.g., PLAYER1 or PLAYER2).
    :return: True if the player has four connected pieces, False otherwise.
    """
    rows, cols = board.shape

    # Check horizontal lines.
    for row in range(rows):
        for col in range(cols - 3):  # Only check up to the fourth-to-last column.
            if np.array_equal(board[row, col:col + 4], [player] * 4):  # Compare slice directly with player value.
                return True

    # Check vertical lines.
    for col in range(cols):
        for row in range(rows - 3):  # Only check up to the fourth-to-last row.
            if np.array_equal(board[row:row + 4, col], [player] * 4):  # Same logic as horizontal
                return True

    # Check diagonal lines (top-left to bottom-right).
    for row in range(rows - 3):
        for col in range(cols - 3):  # Only check up to the fourth-to-last column.
            if np.array_equal(board[row:row + 4, col:col + 4].diagonal(), [player] * 4):
                return True

    # Check diagonal lines (bottom-left to top-right).
    for row in range(3, rows):
        for col in range(cols - 3):  # Only check up to the fourth-to-last column.
            if np.array_equal(np.fliplr(board[row - 3:row + 1, col:col + 4]).diagonal(), [player] * 4):
                return True

    return False



def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Determines the current state of the game based on the provided board and player.

    :param board: A 2D array representing the game board.
    :param player: The identifier of the player to check for a win condition.

    Returns:
    - GameState: The current game state:
        - GameState.IS_WIN: If the player has connected four pieces.
        - GameState.IS_DRAW: If the board is completely filled with no winner.
        - GameState.STILL_PLAYING: If no end condition is met.
    """
    # Check if the given player has achieved a win condition (four connected pieces).
    if connected_four(board, player):
        return GameState.IS_WIN

    # Check if the board is completely filled (no empty spaces remain).
    if np.all(board != NO_PLAYER):
        return GameState.IS_DRAW

    # If no win or draw condition is met, the game is still ongoing.
    return GameState.STILL_PLAYING


def valid_moves(board):
    """Returns a list of valid column indices where a move can be made."""
    return [move for move in range(BOARD_COLS) if check_move_status(board, move) == MoveStatus.IS_VALID]


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:
    """
    Checks the validity of a move in a Connect Four game.

    :param board: A 2D array representing the game board.
    :param column: The column where the move is being attempted. Expected to be a number.

    Returns:
    - MoveStatus: An enumeration representing the move status:
        - MoveStatus.WRONG_TYPE: If the input is not a valid column index.
        - MoveStatus.OUT_OF_BOUNDS: If the column index is outside the board's range.
        - MoveStatus.FULL_COLUMN: If the column is already full.
        - MoveStatus.IS_VALID: If the move is valid and can be executed.
    """
    # Check if the input column is None or an empty string
    if column is None or column == "":
        return MoveStatus.WRONG_TYPE

    try:
        # Convert the input to an integer (handling strings like '3.0')
        col = int(float(column))
    except ValueError:
        # If conversion fails, the input is invalid
        return MoveStatus.WRONG_TYPE

    # Check if the column index is within the board's bounds
    if col < 0 or col >= board.shape[1]:
        return MoveStatus.OUT_OF_BOUNDS

    # Check if the column is full (top row is occupied)
    if board[BOARD_ROWS - 1, col] != NO_PLAYER:
        return MoveStatus.FULL_COLUMN

    # If all checks pass, the move is valid
    return MoveStatus.IS_VALID


def find_occurrences(string, searched_letter):
    """
    Searches for a specific character in a string and gives back all the location of it.

    :param string: String where I search in
    :param searched_letter: character I search in
    :return: A list with the positions
    """
    return [i for i, letter in enumerate(string) if letter == searched_letter]
