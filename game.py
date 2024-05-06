# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function

import os

import numpy as np

import experiment

import json

import time


class TextColor:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    UNDERLINE = '\033[4m'
    GREY = "\033[90m"
    RESET = '\033[0m'


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            # Represents the moves of the current player.
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            # Represents the moves of the opponent.
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            # Indicates which player's turn it is to play next.
            square_state[3][:, :] = 1.0
        #  reverse the order of rows in each matrix to switch coordinate
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        # print("has a winner for fiar?")
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))

        # means that there aren't enough moves on the board to determine a winner.
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            #  convert a one-dimensional index (m) into two-dimensional coordinates (h for height/row and w for width/column)
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def has_a_winner_knobby(self):
        # print("has a winner for knobby?")
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))

        # means that there aren't enough moves on the board to determine a winner.
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            #  convert a one-dimensional index (m) into two-dimensional coordinates (h for height/row and w for width/column)
            h = m // width
            w = m % width
            player = states[m]
            # TODO: Check if there are issues with the following code
            # check for a horizontal bottom 3 in a knob

            # look for the starting position of the base
            if (w in range(width - 2) and
                    len(set(states.get(i, -1) for i in range(m, m + 3))) == 1):  #If i is not a valid key in states (meaning no player has moved there), the method returns -1 by default.
                # check for the middle 2 in a knobby
                # up direction
                vertical_knob_up = set()
                vertical_knob_up.add(states.get(m + 1, -1))
                vertical_knob_up.add(states.get(m + 1 + width, -1))
                vertical_knob_down = set()
                vertical_knob_down.add(states.get(m + 1, -1))
                vertical_knob_down.add(states.get(m + 1 - width, -1))
                if (len(set(vertical_knob_up)) == 1):
                    return True, player
                elif (len(set(vertical_knob_down)) == 1):
                    return True, player

            # check for a vertical bottom 3 in a knob
            if (h in range(height - 2) and
                    len(set(states.get(i, -1) for i in range(m, m + 3 * width, width))) == 1):
                # check for the middle 2 in a knobby
                horizontal_knob_left = set()
                horizontal_knob_right = set()

                # account for m at the edge of the board
                # if m is at the right edge, then the right knob is not valid
                # if m is at the left edge, then the left knob is not valid
                if (w in range(width - 1)):
                    horizontal_knob_right.add(states.get(m + width, -1))
                    horizontal_knob_right.add(states.get(m + width + 1, -1))
                else:
                    horizontal_knob_right.add(-1)
                if (w in range(1, width)):
                    horizontal_knob_left.add(states.get(m + width, -1))
                    horizontal_knob_left.add(states.get(m + width - 1, -1))
                else:
                    horizontal_knob_left.add(-1)

                if (len(set(horizontal_knob_right)) == 1):
                    return True, player
                elif (len(set(horizontal_knob_left)) == 1):
                    return True, player

        return False, -1

    def game_end(self, m=2):
        """Check whether the game is ended or not"""
        global params
        params = load_params_from_file()
        if params is None and m == 2:
            # go to an upper directory
            parent_dir = os.path.dirname(os.getcwd())
            params = load_params_from_file(parent_dir + "/params.json")
            m = params["model"]
        if m == 0:
            win, winner = self.has_a_winner()
        elif m == 1:
            win, winner = self.has_a_winner_knobby()
        else:
            print("Invalid model type")
            return False, -1

        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board
        self.game_steps = 0
        self.player_steps = 0
        self.game_state = np.empty((300, 4, 6, 6), dtype=object)
        self.player_rt = np.empty((300, 1), dtype=object)
        self.prob_knobby = np.empty((300, 6, 6), dtype=object)
        self.prob_fouriar = np.empty((300, 6, 6), dtype=object)
        self.prob_knobby_full = np.empty((300, 6, 6), dtype=object)
        self.prob_fouriar_full = np.empty((300, 6, 6), dtype=object)
        self.board_differnece = np.empty((300, 2, 6, 6), dtype=object)

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X (You)".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print(TextColor.BLUE + "{0:8}".format(x) + TextColor.RESET, end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print(TextColor.YELLOW + "{0:4d}".format(i) + TextColor.RESET, end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)

                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')

            # Print row number on the right side followed by a newline
            print(TextColor.YELLOW + "{0:4d}".format(i) + TextColor.RESET, end='\r\n\r\n')

        for x in range(width):
            print(TextColor.BLUE + "{0:8}".format(x) + TextColor.RESET, end='')
        print('\r\n')

    def start_play(self, player1, player2, start_player=1, is_shown=1):
        """start a game between two players"""
        global params
        params = load_params_from_file()

        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')

        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            # Get MCTS probabilities for the current board state.

            # 1 -- Experiment Only --
            # if current_player == 1:
            #     move, move_probs_fiar, move_probs_knobby, rt = player_in_turn.get_action(self.board,
            #                                                                              return_prob=2,
            #                                                                              return_rt=1)
            # else:
            #     move, move_probs = player_in_turn.get_action(self.board,
            #                                                  return_prob=1,
            #                                                  temp=0.85)


            # 2 -- Tain Model Only ---
            move = player_in_turn.get_action(self.board)
            # --

            self.board.do_move(move)
            self.game_steps += 1

            # 1 -- Experiment Only --
            # if current_player == 1:
            #     self.player_steps += 1
            #     if params["model"] == 0:
            #         params["moves_fouriar"] -= 1
            #     elif params["model"] == 1:
            #         params["moves_knobby"] -= 1
            #     store_params_to_file()
            #
            #     print("Player 1 turn")
            #     # Save the board matrix
            #     move_probs_fiar_reshaped = move_probs_fiar.reshape(self.board.width, self.board.height)
            #     move_probs_knobby_reshaped = move_probs_knobby.reshape(self.board.width, self.board.height)
            #
            #     state_matrices = self.board.current_state()
            #     player_moves = state_matrices[1]
            #     ai_moves = state_matrices[0]
            #     ai_moves[ai_moves == 1] = 2
            #     last_move = state_matrices[2]
            #     previous_matrix = player_moves + ai_moves - last_move
            #     current_matrix = player_moves + ai_moves
            #     mask = last_move != 0
            #
            #     # print("last move\n", last_move)
            #
            #     # Use this mask to capture the prob only at the last-move location
            #     # then, flip the rows to match the coordinate system
            #     singleMove_probs_fiar = np.zeros_like(move_probs_fiar_reshaped)
            #     singleMove_probs_fiar[mask] = move_probs_fiar_reshaped[mask]
            #     # singleMove_probs_fiar = np.flip(singleMove_probs_fiar, 0)
            #     singleMove_probs_knobby = np.zeros_like(move_probs_knobby_reshaped)
            #     singleMove_probs_knobby[mask] = move_probs_knobby_reshaped[mask]
            #     # singleMove_probs_knobby = np.flip(singleMove_probs_knobby, 0)
            #
            #     # print("prob fiar", singleMove_probs_fiar)
            #     # print("prob knobby", singleMove_probs_knobby)
            #
            #     # --
            #     # Get MCTS probabilities for the current board state.
            #     # round
            #     # reverse order of rows to switch coordinate system
            #     # prob_matrix_fiar = singleMove_probs_fiar.reshape(self.board.width, self.board.height)
            #     # prob_matrix_fiar = np.flip(singleMove_probs_fiar, 0)
            #
            #     # reverse order of rows to switch coordinate system
            #     # prob_matrix_knobby = singleMove_probs_knobby.reshape(self.board.width, self.board.height)
            #     # prob_matrix_knobby = np.flip(singleMove_probs_knobby, 0)
            #     # --
            #
            #     # save above data to one file
            #     # self.game_state[self.player_steps-1] = np.array([prob_matrix, previous_matrix, last_move, current_matrix])
            #
            #     self.board_differnece[self.player_steps - 1] = np.array([previous_matrix, current_matrix])
            #     self.prob_knobby[self.player_steps - 1] = singleMove_probs_knobby
            #     self.prob_fouriar[self.player_steps - 1] = singleMove_probs_fiar
            #     self.prob_knobby_full[self.player_steps - 1] = move_probs_knobby_reshaped
            #     self.prob_fouriar_full[self.player_steps - 1] = move_probs_fiar_reshaped
            #     self.player_rt[self.player_steps - 1] = rt
            #
            #     experiment.save_game_data(self.prob_fouriar_full, "fullProbFiar")
            #     experiment.save_game_data(self.prob_knobby_full, "fullProbKnobby")
            #
            #     experiment.save_game_data(self.prob_knobby, "probKnobby")
            #     experiment.save_game_data(self.prob_fouriar, "probFouriar")
            #     experiment.save_game_data(self.board_differnece, "boardDifference", False)
            #     experiment.save_game_data(self.player_rt, "RT")
            #
            # --

            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print(TextColor.CYAN + "Game end. Winner is " + str(players[winner]) + TextColor.RESET)
                        print()
                        # 1. Experiment Only
                        # params["games_rule"].append(params["model"]) # 0: fiar, 1: knobby
                        # params["games_results"].append(winner)
                        # store_params_to_file()
                        # experiment.update_with_condition()
                    else:
                        print(TextColor.CYAN + "Game end. No Winner. Tie" + TextColor.RESET)
                        print()
                        # 1. Experiment Only
                        # params["games_rule"].append(params["model"])  # 0: fiar, 1: knobby
                        # params["games_results"].append(3)
                        # store_params_to_file()
                        # experiment.update_with_condition()
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        global params
        params = load_params_from_file()

        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1) # training knobby model
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print(TextColor.CYAN + "Game end. Winner is player: " + str(winner) + TextColor.RESET)
                        params["state"] = 2
                        store_params_to_file()
                    else:
                        print(TextColor.CYAN + "Game end. No winner. Tie" + TextColor.RESET)
                        params["state"] = 2
                        store_params_to_file()
                return winner, zip(states, mcts_probs, winners_z)


def store_params_to_file(filename="params.json"):
    if not os.path.exists(filename):
        return None
    with open(filename, 'w') as file:
        json.dump(params, file)


def load_params_from_file(filename="params.json"):
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as file:
        return json.load(file)
