import pytest
import numpy as np

from game_utils import initialize_game_state, BOARD_SHAPE, BoardPiece, pretty_print_board, \
    clear_unwanted_characters, string_to_board, NO_PLAYER, PLAYER1, PLAYER2, connected_four


def test_initialize_game_state():
    board = initialize_game_state()
    assert board.shape == BOARD_SHAPE
    assert board.dtype == BoardPiece
    assert np.all(board == NO_PLAYER)


def test_string_to_board():
    board_as_str = ('|==============|\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|  X       O   |\n'
                    '|==============|\n'
                    '|0 1 2 3 4 5 6 |\n')

    # Erstelle ein erwartetes Board-Array zur Verifikation
    expected_board = initialize_game_state()
    expected_board[0, 1] = PLAYER1
    expected_board[0, 5] = PLAYER2

    # Teste die Konvertierung
    board = string_to_board(board_as_str)
    assert np.array_equal(board,
                          expected_board), "Das konvertierte Board stimmt nicht mit dem erwarteten Board überein."

    # Teste, ob pretty_print_board korrekt das erwartete Board-Format ausgibt
    assert pretty_print_board(board) == board_as_str


def test_clear_unwanted_characters():
    board_as_str = ('|==============|\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|              |\n'
                    '|  X       O   |\n'
                    '|==============|\n'
                    '|0 1 2 3 4 5 6 |\n')
    assert clear_unwanted_characters(board_as_str) == ('       \n'
                                                       '       \n'
                                                       '       \n'
                                                       '       \n'
                                                       '       \n'
                                                       ' X   O \n')


def test_connected_four():
    board = initialize_game_state()

    # Test horizontal connection
    board[0, 0] = PLAYER1
    board[0, 1] = PLAYER1
    board[0, 2] = PLAYER1
    board[0, 3] = PLAYER1
    assert connected_four(board, PLAYER1) == True, "Horizontal connection failed"

    # Test vertical connection
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[1, 0] = PLAYER1
    board[2, 0] = PLAYER1
    board[3, 0] = PLAYER1
    assert connected_four(board, PLAYER1) == True, "Vertical connection failed"

    # Test diagonal connection (top-left to bottom-right)
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[1, 1] = PLAYER1
    board[2, 2] = PLAYER1
    board[3, 3] = PLAYER1
    assert connected_four(board, PLAYER1) == True, "Diagonal (TL-BR) connection failed"

    # Test diagonal connection (bottom-left to top-right)
    board = initialize_game_state()
    board[5, 0] = PLAYER1
    board[4, 1] = PLAYER1
    board[3, 2] = PLAYER1
    board[2, 3] = PLAYER1
    assert connected_four(board, PLAYER1) == True, "Diagonal (BL-TR) connection failed"

    # Test no connection
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[0, 1] = PLAYER2
    board[0, 2] = PLAYER1
    board[0, 3] = PLAYER2
    assert connected_four(board, PLAYER1) == False, "No connection (horizontal) failed"
