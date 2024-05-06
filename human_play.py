# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
@modified by Hannah Zeng
"""

from __future__ import print_function

import numpy as np

import random
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
import pickle


import json

import time
import os

# weaker_four_model = "best_policy_6_6_4_1010_mid.model"
# weaker_knobby_model = "best_policy_6_6_knobby_1011_mid.model"
four_model = "best_policy_6_6_4_1008.model"
knobby_model = "best_policy_6_6_knobby_1008.model"
best_four_model = "best_policy_6_6_4_1008.model"
best_knobby_model = "best_policy_6_6_knobby_1008.model"
new_knobby_model = "best_policy_6_6_k_0311.model"

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


class Human(object):
    """
    human player
    """

    def __init__(self):
        global params
        params = load_params_from_file()
        self.player = None
        self.mcts_hidden = None
        if params is not None:
            if params["model"] == 0:
                self.model = four_model
            elif params["model"] == 1:
                self.model = knobby_model
            else:
                self.model = four_model

    def set_player_ind(self, p):
        self.player = p

    def set_hidden_player(self, board, m=0):
        if m == 0:
            model = best_four_model
        elif m == 1:
            # model = best_knobby_model
            model = new_knobby_model
        elif m == 2: #?????
            params = load_params_from_file()
            if params["model"] == 0:
                model = best_four_model
            elif params["model"] == 1:
                model = new_knobby_model
            #     model = best_knobby_model

        best_policy = PolicyValueNet(board.width, board.height, model_file=model)
        self.mcts_hidden = MCTSPlayer(best_policy.policy_value_fn,
                                      c_puct=5,
                                      n_playout=1000)

    def get_hidden_probability(self, board, temp):
        move_probs = np.zeros(board.width * board.height)
        acts, probs = self.mcts_hidden.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        # check if the board is zero
        if np.sum(move_probs) == 0:
            print("get_hidden_probability: move_probs is zero")
        return move_probs

    # 1 -- Experiment Only --
    # if current_player == 1:
    #     move, move_probs_fiar, move_probs_knobby, rt = player_in_turn.get_action(self.board,
    #                                                                              return_prob=2,
    #                                                                              return_rt=1)
    # else:
    #     move, move_probs = player_in_turn.get_action(self.board,
    #                                                  return_prob=1,
    #                                                  temp=0.85)
    def get_action(self, board, temp=0.75, return_prob=0, return_rt=0):
        # temp (0, 1] needs to be larger for detailed probability map

        # --- get the action from a MCTS player for evaluating player move

        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        move_probs_fiar = np.zeros(board.width * board.height)
        move_probs_knobby = np.zeros(board.width * board.height)

        if len(sensible_moves) > 0:
            if return_prob == 2: #return both probabilities
                self.set_hidden_player(board, 0)
                move_probs_fiar = self.get_hidden_probability(board, 0.75)
                self.set_hidden_player(board, 1)
                move_probs_knobby = self.get_hidden_probability(board, 0.75)
            elif return_prob == 1:
                self.set_hidden_player(board, 2)
                move_probs = self.get_hidden_probability(board, temp)
        else:
            move_probs_fiar = None
            move_probs_knobby = None
            move_probs = None

        # ---

        try:
            rt_start = time.time()
            location = input(
                "Your move, type "
                + TextColor.YELLOW + "row"
                + TextColor.RESET + ","
                + TextColor.BLUE + "column"
                + TextColor.RESET + ": ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
            rt_end = time.time()
            rt = rt_end - rt_start

        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move, please try again.")
            # move = self.get_action(board)
            move, move_probs_fiar, move_probs_knobby, rt = self.get_action(board,
                                                                             return_prob=2,
                                                                             return_rt=1)

        # allow human player get actions to return probabilities too
        if return_prob == 2 and return_rt == 1:
            return move, move_probs_fiar, move_probs_knobby, rt
        elif return_prob == 1:
            return move, move_probs
        else:
            return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    params = load_params_from_file()

    n = 4
    width, height = 6, 6
    if params["model"] == 0:
        model_file = four_model
    elif params["model"] == 1:
        model_file = knobby_model


    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        """
        param_theano = pickle.load(open(model_file, 'rb'))
        keys = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias'
            , 'act_conv1.weight', 'act_conv1.bias', 'act_fc1.weight', 'act_fc1.bias'
            , 'val_conv1.weight', 'val_conv1.bias', 'val_fc1.weight', 'val_fc1.bias', 'val_fc2.weight', 'val_fc2.bias']
        param_pytorch = OrderedDict()
        for key, value in zip(keys, param_theano):
            if 'fc' in key and 'weight' in key:
                param_pytorch[key] = torch.FloatTensor(value.T)
            elif 'conv' in key and 'weight' in key:
                param_pytorch[key] = torch.FloatTensor(value[:, :, ::-1, ::-1].copy())
            else:
                param_pytorch[key] = torch.FloatTensor(value)
        """
        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        # best_policy = PolicyValueNet(width, height, model_file = model_file)
        # mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        try:
            policy_param = model_file
            # policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        best_policy = PolicyValueNet(width, height, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first, =1 for AI first
        random.seed(time.time())
        player_sequence = random.choice([0, 1])
        game.start_play(human, mcts_player, start_player=player_sequence, is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')

def load_params_from_file(filename="params.json"):
    # check if the file exists
    if not os.path.exists(filename):
        return None
    with open(filename, 'r') as file:
        return json.load(file)


if __name__ == '__main__':
    run()
