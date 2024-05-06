# Loop through the board difference files of each participant
# and get the board probability maps for each move
#
from collections import defaultdict
from mcts_alphaZero import MCTSPlayer, softmax
import os
import numpy as np
from game import Board, Game
from human_play import Human

player = Human()
board = Board(width=6, height=6, n_in_row=4)
game = Game(board)


temp = 0.75
move_probs_fiar = np.zeros(6 * 6)
move_probs_knobby = np.zeros(6 * 6)
root = os.path.dirname(os.path.realpath(__file__))

def state_to_prob(matrix):
    # board states stored as a dict,
    # - key: move as location on the board,
    # - value: player as pieces type
    # print(matrix)
    state = defaultdict()
    for i in range(6):
        for j in range(6):
            if matrix[1][i, j] == 2:
                m = board.location_to_move([i, j])
                state[m] = 2 # player 2 is the AI player
            elif matrix[1][i, j] == 1:
                m = board.location_to_move([i, j])
                state[m] = 1 # player 1 is the human player
    last_m = matrix[1] - matrix[0]
    last_m = np.where(last_m == 1) # [h],[w] = (array([h], dtype=int64), array([w], dtype=int64))
    # location to matrix of boolean
    mask = np.zeros((6, 6))
    mask[last_m[0][0], last_m[1][0]] = 1
    mask = mask.astype(bool)
    board.last_move = board.location_to_move([last_m[0][0], last_m[1][0]])

    return get_board_prob_maps(state, mask)

def get_board_prob_maps(state, mask=None):
    global temp

    board.states = state
    board.current_player = 1
    # get availables from board's empty position
    board.availables = range(6 * 6)
    board.availables = list(filter(lambda x: x not in board.states.keys(), board.availables))
    board.availables.append(board.last_move)

    print("=============FIAR")
    # Check if the game has ended for Four-in-a-row
    end_fiar, winner_fiar = board.game_end(m=0)
    if end_fiar:
        print("Four-in-a-row game has ended. Winner:", winner_fiar)
    player.set_hidden_player(board, 0)
    full_move_prob_fiar = player.get_hidden_probability(board, temp)

    print("=============KNOBBY")
    # Check if the game has ended for Knobby
    # end_knobby, winner_knobby = board.game_end(m=1)
    # if end_knobby:
    #     print("Knobby game has ended. Winner:", winner_knobby)
    # player.set_hidden_player(board, 1)
    # full_move_prob_knobby = player.get_hidden_probability(board, temp)

    # from move to matrix
    full_move_prob_fiar = np.reshape(full_move_prob_fiar, (6, 6))
    # full_move_prob_knobby = np.reshape(full_move_prob_knobby, (6, 6))

    # return full_move_prob_fiar, full_move_prob_knobby
    return full_move_prob_fiar

def threshold_matrix(matrix, threshold):

    modified_matrix = np.where(matrix < threshold, threshold, matrix)
    return modified_matrix

def threshold_matrices(matrices, threshold=1e-10):

    # Apply the threshold_matrix function to each matrix in the list
    modified_matrices = [threshold_matrix(matrix, threshold) for matrix in matrices]
    return modified_matrices
