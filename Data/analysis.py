import math
import os
import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from experiment import save_game_data_simple
from simulation import state_to_prob, threshold_matrices
import seaborn as sns

# Set the global font
plt.rcParams['font.family'] = ['Arial', 'sans-serif']

deprecated_id = ['p03', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p35', 'p16', 'p18', 'p13']
# p35 - problematic params
# p16 - lose params
params_list = []
paths_blocked = []
paths_interleaved = []
id_blocked = []
id_interleaved = []

win_rate_blocked_1 = []
win_rate_interleaved_1 = []
win_rate_blocked_2 = []
win_rate_interleaved_2 = []
win_rate_knobby_all = []
win_rate_fiar_all = []

aggregated_y1 = []
aggregated_y2 = []

root = os.path.dirname(os.path.realpath(__file__))
colors = sns.color_palette('Paired')
c_blue_1 = colors[0]  # lighter
c_blue_2 = colors[1]  # darker
c_green_1 = colors[2]
c_green_2 = colors[3]
c_red_1 = colors[4]
c_red_2 = colors[5]
c_orange_1 = colors[6]
c_orange_2 = colors[7]
c_purple_1 = colors[8]
c_purple_2 = colors[9]
c_black = 'black'


# return
# 1) the prob number
# 2) the move position

def find_duplicate_params():
    global root

    params_dict = defaultdict(list)
    temp_root = root
    for temp_root, dirs, files in os.walk(temp_root):
        for file_name in files:
            if file_name == 'params.json':
                params_path = os.path.join(temp_root, file_name)
                # Extract participant ID from the file path
                participant_id = os.path.basename(os.path.dirname(params_path))
                if (participant_id not in deprecated_id):
                    params_dict[participant_id].append(params_path)

    return params_dict


# group data by condition (blocked=0 vs interleaved=1)
def group_by_condition(params_list):
    global paths_blocked, paths_interleaved, id_blocked, id_interleaved

    for id, file_paths in params_list.items():
        # print("id", id, "file_paths", file_paths)
        for path in file_paths:
            with open(path, 'r') as file:
                params_data = json.load(file)

            condition = params_data.get('condition', 0)
            this_id = params_data.get('participant_id', 0)
            if condition == 0:
                paths_blocked.append(path)
                id_blocked.append(this_id)
            else:
                paths_interleaved.append(path)
                id_interleaved.append(this_id)
    print("blocked ids", id_blocked, "\ninterleaved ids", id_interleaved)


def get_game_sequence_by_id(id):
    # find corresponding params file
    for file_name, file_paths in params_list.items():
        for path in file_paths:
            with open(path, 'r') as file:
                params_data = json.load(file)
            if (params_data['participant_id'] == id):
                first_game = params_data.get('games_rule', [])[0]
                games_rule = params_data.get('games_rule', [])
                # get the original game type Fiar (0) or Knobby (1)
                # turn all games into isFirstGame (0) or not (1)
                for i in range(len(games_rule)):
                    if games_rule[i] == first_game:
                        games_rule[i] = 0
                    else:
                        games_rule[i] = 1
                return first_game, games_rule


# return
# 1) the first game rule played
# 2) the results of the first game
# 3) the results of the second game
def calculate_win(params_path):
    with open(params_path, 'r') as file:
        params_data = json.load(file)

    if (params_data['participant_id'] in deprecated_id):
        return None, None, None

    # Extract relevant information
    id = params_data['participant_id']
    games_rule = params_data.get('games_rule', [])
    games_results = params_data.get('games_results', [])

    if games_rule == []:  #
        print("No games_rule for", id)
    # which game is played first
    first_game = games_rule[0]

    # List of all win games
    results_four = []
    results_knobby = []

    # Seperate results into two lists
    for result, rule in zip(games_results, games_rule):
        if rule == 0:  # Four-in-a-row
            results_four.append(result)
        elif rule == 1:  # Knobby
            results_knobby.append(result)

    if (first_game == 0):
        return id, first_game, results_four, results_knobby
    else:
        return id, first_game, results_knobby, results_four


# return the win rate for all games
# given the cumulative results
def get_win_rate_all(data):
    win_rate = []
    sum_win = [0] * 60
    count = [0] * 60
    n = len(data)
    for id, list in data.items():
        result = list
        result = [0 if r != 1 else 1 for r in result]

        for j in range(len(result)):  # j = game number
            sum_win[j] += result[j]
            count[j] += 1
    while count and count[-1] == 0:
        count.pop()
    # max = count[0] # this is usually ~13
    max = 9
    set_in_count = set(count)
    for i in range(len(set_in_count)):
        if count[i] < max:
            max = count[i - 1]
    count = [x for x in count if x >= max]  # make sure each point has at least 8 data points
    sum_win = sum_win[:len(count)]  # slice to the same length as count

    for i in range(len(sum_win)):
        if count[i] != 0:
            w = round(sum_win[i] / count[i], 3)
            win_rate.append(w)

    return win_rate, max


def get_win_rate_by_order(dataframe):
    # return win rates across participants by 'play_order' column
    win_rate_player_first = []
    win_rate_player_second = []
    win_rate_player_first.append(0)  # first round is always 0
    win_rate_player_second.append(0)  # first round is always 0

    games_per_round = {}
    win_per_round = {}

    df_player_first = dataframe[dataframe['play_order'] == 1].copy()
    df_player_second = dataframe[dataframe['play_order'] == 0].copy()

    for df in [df_player_first, df_player_second]:
        for game_round in df['round'].unique():
            print(f"Round: {game_round}")

            # get this round data that are first moves
            df_this_round = df[(df['round'] == game_round) & (df['is_first_move'] == True)].copy()

            # check why list index out of range
            print("len of df_this_round", len(df_this_round))

            games_per_round[game_round] = games_per_round.get(game_round, 0) + len(df_this_round)
            win_per_round[game_round] = win_per_round.get(game_round, 0) + sum(df_this_round['result'])

        sorted_rounds = sorted(games_per_round.keys())

        for round_num in sorted_rounds:
            if games_per_round[round_num] != 0 and win_per_round[round_num] != 0:
                w = round(win_per_round[round_num] / games_per_round[round_num], 3)
                if df is df_player_first:
                    win_rate_player_first.append(w)
                else:
                    win_rate_player_second.append(w)

    print("win_rate_player_first", win_rate_player_first)
    print("win_rate_player_second", win_rate_player_second)

    return win_rate_player_first, win_rate_player_second


def plot_win_rate(count, df):
    global win_rate_blocked_1, win_rate_interleaved_1, win_rate_blocked_2, win_rate_interleaved_2, \
        win_rate_fiar_all, win_rate_knobby_all

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True, dpi=300)
    plt.tight_layout(pad=3.0)
    fig_both, (ax3, ax4) = plt.subplots(1, 2, figsize=(8, 4), sharey=True, dpi=300)
    fig_playOrder, (ax5, ax6) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    plt.tight_layout(pad=3.0)

    win_rate_player_first, win_rate_player_second = get_win_rate_by_order(df)

    # unpack count of blocked / interleaved + fiar / knobby
    (blocked_fiar,
     blocked_knobby,
     interleaved_fiar,
     interleaved_knobby) = count
    blocked_fiar = int(blocked_fiar[0])
    blocked_knobby = int(blocked_knobby[0])
    interleaved_fiar = int(interleaved_fiar[0])
    interleaved_knobby = int(interleaved_knobby[0])

    avg_win_fiar = sum(win_rate_fiar_all) / len(win_rate_fiar_all)
    avg_win_knobby = sum(win_rate_knobby_all) / len(win_rate_knobby_all)

    avg_win_player_first = sum(win_rate_player_first) / len(win_rate_player_first)
    avg_win_player_second = sum(win_rate_player_second) / len(win_rate_player_second)

    # Create x-axis
    x1_1 = list(range(1, len(win_rate_blocked_1) + 1))
    x2_1 = list(range(1, len(win_rate_interleaved_1) + 1))
    x1_2 = list(range(1, len(win_rate_blocked_2) + 1))
    x2_2 = list(range(1, len(win_rate_interleaved_2) + 1))

    x3 = list(range(1, len(win_rate_fiar_all) + 1))
    x4 = list(range(1, len(win_rate_knobby_all) + 1))
    x5 = list(range(1, len(win_rate_player_first) + 1))
    x6 = list(range(1, len(win_rate_player_second) + 1))

    print("len of x1_1", len(x1_1))
    print("len of x2_1", len(x2_1))
    print("len of x1_2", len(x1_2))
    print("len of x2_2", len(x2_2))
    print("len of x3", len(x3))
    print("len of x4", len(x4))
    print("len of x5", len(x5))
    print("len of x6", len(x6))

    # number to fraction
    # x1_1 = np.linspace(0, 1, len(win_rate_blocked_1))
    # x2_1 = np.linspace(0, 1, len(win_rate_interleaved_1))
    # x1_2 = np.linspace(0, 1, len(win_rate_blocked_2))
    # x2_2 = np.linspace(0, 1, len(win_rate_interleaved_2))

    # Fitting polynomial
    poly_1_1 = np.poly1d(np.polyfit(x1_1, win_rate_blocked_1, 3))
    smooth_b1 = poly_1_1(x1_1)
    poly_2_1 = np.poly1d(np.polyfit(x2_1, win_rate_interleaved_1, 3))
    smooth_i1 = poly_2_1(x2_1)
    poly_1_2 = np.poly1d(np.polyfit(x1_2, win_rate_blocked_2, 3))
    smooth_b2 = poly_1_2(x1_2)
    poly_2_2 = np.poly1d(np.polyfit(x2_2, win_rate_interleaved_2, 3))
    smooth_i2 = poly_2_2(x2_2)
    poly_3 = np.poly1d(np.polyfit(x3, win_rate_fiar_all, 3))
    smooth_fiar = poly_3(x3)
    poly_4 = np.poly1d(np.polyfit(x4, win_rate_knobby_all, 3))
    smooth_knobby = poly_4(x4)
    poly_5 = np.poly1d(np.polyfit(x5, win_rate_player_first, 3))
    smooth_player_first = poly_5(x5)
    poly_6 = np.poly1d(np.polyfit(x6, win_rate_player_second, 3))
    smooth_player_second = poly_6(x6)

    # Histograms for all fiar and knobby games
    # ax5.hist(win_rate_fiar_all, bins=10, color=c_green_1, alpha=0.7)
    # ax5.set_title('Fiar games')
    # ax6.hist(win_rate_knobby_all, bins=10, color=c_purple_1, alpha=0.7)
    # ax6.set_title('Knobby games')

    intercept_b1 = poly_1_1.coefficients[3]
    intercept_i1 = poly_2_1.coefficients[3]
    intercept_b2 = poly_1_2.coefficients[3]
    intercept_i2 = poly_2_2.coefficients[3]

    label_b1 = f'Intercept (first game): {intercept_b1:.3f}'
    label_i1 = f'Intercept (first game): {intercept_i1:.3f}'
    label_b2 = f'Intercept (second game): {intercept_b2:.3f}'
    label_i2 = f'Intercept (second game): {intercept_i2:.3f}'

    # Blocked condition
    ax1.scatter(x1_1, win_rate_blocked_1, color=c_green_1, alpha=0.7, s=20, label=label_b1)
    ax1.plot(x1_1, smooth_b1, color=c_green_2)
    ax1.scatter(x1_2, win_rate_blocked_2, color=c_purple_1, alpha=0.7, s=20, label=label_b2)
    ax1.plot(x1_2, smooth_b2, color=c_purple_2)

    # Interleaved condition
    ax2.scatter(x2_1, win_rate_interleaved_1, color=c_green_1, alpha=0.7, s=20, label=label_i1)
    ax2.plot(x2_1, smooth_i1, color=c_green_2)
    ax2.scatter(x2_2, win_rate_interleaved_2, color=c_purple_1, alpha=0.7, s=20, label=label_i2)
    ax2.plot(x2_2, smooth_i2, color=c_purple_2)

    # All fiar games
    ax3.scatter(x3, win_rate_fiar_all, color=c_black, alpha=0.3, s=20)
    ax3.plot(x3, smooth_fiar, color=c_black)
    ax3.plot(x3, [avg_win_fiar] * len(x3), linestyle='--', color=c_black, label=f'Average: {avg_win_fiar:.3f}')

    # All knobby games
    ax4.scatter(x4, win_rate_knobby_all, color=c_black, alpha=0.3, s=20)
    ax4.plot(x4, smooth_knobby, color=c_black)
    ax4.plot(x4, [avg_win_knobby] * len(x4), linestyle='--', color=c_black, label=f'Average: {avg_win_knobby:.3f}')

    # Player order
    ax5.scatter(x5, win_rate_player_first, color=c_black, alpha=0.3, s=20)
    ax5.plot(x5, smooth_player_first, color=c_black)
    ax5.plot(x5, [avg_win_player_first] * len(x5), linestyle='--', color=c_black, label=f'Average: {avg_win_player_first:.3f}')
    ax6.scatter(x6, win_rate_player_second, color=c_black, alpha=0.3, s=20)
    ax6.plot(x6, smooth_player_second, color=c_black)
    ax6.plot(x6, [avg_win_player_second] * len(x6), linestyle='--', color=c_black, label=f'Average: {avg_win_player_second:.3f}')

    # Add title and labels
    ax1.set_title(f'Blocked (n=13)\nFirst game: {blocked_fiar} Fiar, {blocked_knobby} Knobby')
    ax2.set_title(f'Interleaved (n=13)\nFirst game: {interleaved_fiar} Fiar, {interleaved_knobby} Knobby')
    ax3.set_title(f'All Four-in-a-row games')
    ax4.set_title(f'All Knobby games')
    ax5.set_title(f'Human goes first')
    ax6.set_title(f'AI goes first')

    for ax in [ax1, ax3, ax5]:
        ax.set_xlabel('Game Rounds')
        ax.set_ylabel('Win rate')
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.legend(loc='upper right')

    ax1.set_xlim(0, 21)
    ax1.set_xticks(np.arange(0, 21, 5))
    ax2.set_xlim(0, 26)
    ax2.set_xticks(np.arange(0, 26, 5))
    ax3.set_xlim(0, 26)
    ax3.set_xticks(np.arange(0, 26, 5))
    ax3.set_yticks(np.arange(0, 1, 0.1))
    ax4.set_xlim(0, 26)
    ax4.set_xticks(np.arange(0, 26, 5))
    ax4.set_yticks(np.arange(0, 1, 0.1))
    ax5.set_xlim(0, 56)
    ax5.set_xticks(np.arange(0, 56, 5))
    ax5.set_yticks(np.arange(0, 1, 0.1))
    ax6.set_xlim(0, 56)
    ax6.set_xticks(np.arange(0, 56, 5))
    ax6.set_yticks(np.arange(0, 1, 0.1))

    plt.show()


def get_all_move_prob(id):
    global root
    # print("id", id)
    all_moves_dict = defaultdict(list)
    all_board_dict = defaultdict(list)
    all_prob_dict = defaultdict(list)
    all_prob_dict_new = defaultdict(list)
    all_games_round = []
    all_games_num = []
    all_move_pos_first = defaultdict(list)
    all_move_pos_last = defaultdict(list)
    all_rt = defaultdict(list)
    all_play_order = defaultdict(list)
    all_results = defaultdict(list)

    # get game results
    file_params = "params.json"
    path_param = os.path.join(root, f"{id}\\{file_params}")
    param = json.load(open(path_param))
    games_results = param.get('games_results', [])
    games_results = [0 if r != 1 else 1 for r in games_results]

    # iterate through all games played by the participant
    for round in range(1, 60):  # 60 is the max #games played so far
        # create a dictionary to store the move and prob for each round/game
        move_dict = defaultdict(list)
        prob_dict = defaultdict(list)
        prob_dict_new = defaultdict(list)
        rt_dict = defaultdict(list)
        board_dict = defaultdict(list)
        is_player_first = defaultdict(list)

        i = str(round)
        file_probKnobby = id + "_fullProbKnobby_" + i + ".npy"
        file_probFour = id + "_fullProbFiar_" + i + ".npy"
        file_move = id + "_boardDifference_" + i + ".npy"
        new_file_probKnobby = id + "_new_fullProbKnobby_" + i + ".npy"
        new_file_probFour = id + "_new_fullProbFiar_" + i + ".npy"
        file_rt = id + "_RT_" + i + ".npy"

        # new_file_probKnobby = id + "_new_fullProbKnobby_" + i
        # new_file_probFour = id + "_new_fullProbFiar_" + i

        path_k = os.path.join(root, f"{id}\\{file_probKnobby}")
        path_f = os.path.join(root, f"{id}\\{file_probFour}")
        path_move = os.path.join(root, f"{id}\\{file_move}")
        path_k_new = os.path.join(root, f"{id}\\{new_file_probKnobby}")
        path_f_new = os.path.join(root, f"{id}\\{new_file_probFour}")
        path_rt = os.path.join(root, f"{id}\\{file_rt}")

        # check if the file exists
        if not os.path.exists(path_move):
            continue
        else:
            probKnobby = np.load(path_k, allow_pickle=True)
            probFour = np.load(path_f, allow_pickle=True)
            move = np.load(path_move, allow_pickle=True)
            probKnobby_new = np.load(path_k_new, allow_pickle=True)
            probFour_new = np.load(path_f_new, allow_pickle=True)
            rt = np.load(path_rt, allow_pickle=True)

            move = move[:20]  # all participants finish within 20 moves, so remove redundant 0s
            n_steps = len(move)

            # -------------------------
            # full_fiar = []
            # full_knobby = []
            # check_path_k = path_k_new + ".npy"
            # check_path_f = path_f_new + ".npy"
            # -------------------------

            move_pos_first = defaultdict(list)
            move_pos_last = defaultdict(list)

            # check player order - if the matrix contains 2 in move[0]
            if np.any(move[0] == 2):
                is_player_first[round] = 0
            else:
                is_player_first[round] = 1

            for s in range(n_steps):
                if np.all(move[s][0] == None) or np.all(move[s][1] == None):  # skip if the move is None
                    continue
                # Check if this is the last non-None move in the sequence
                move_pos_first[s] = s == 0
                is_last_move = np.all(move[s + 1] == None)
                move_pos_last[s] = is_last_move

                step = move[s][1] - move[s][0]
                move_dict[s] = step

                probFour[s] = np.flip(probFour[s], 0)  # flip row to match coordinate system
                probKnobby[s] = np.flip(probKnobby[s], 0)  # flip row

                board_dict[s] = move[s][0]  # what is fed into the model
                prob_dict[s] = probFour[s], probKnobby[s]
                prob_dict_new[s] = probFour_new[s], probKnobby_new[s]
                rt_dict[s] = rt[s][0]  # rt[s] is a tuple
                all_games_round.append(round)

                # NEW get new probability matrices if haven't
                # -------------------------
                # if not os.path.exists(check_path_k) or not os.path.exists(check_path_f):
                # print("id", id, "round", round, "step", s)
                # f, k = state_to_prob(move[s]) # for each step
                # full_fiar.append(f) # for each game
                # f = state_to_prob(move[s])
                # full_fiar.append(f)
                # -------------------------
            # print("probs knobby for this round", probKnobby)
            # print("probs fiar for this round", probFour)

            # smoothen probability
            # p_smoothed = (1-α) * p_model + α*(1/(# empty squares))
            α = 0.01
            for s in range(probFour_new.shape[0]):
                # print(probFour_new[s])
                # Define a tolerance level
                tolerance = 1e-50
                # Find indices where elements are not close to zero
                nonzero_indices = np.where(np.abs(probFour_new[s]) > tolerance)
                # Count the number of available elements
                n_available = len(nonzero_indices[0])
                if (n_available == 0):
                    print("No available moves in round", round, "step", s)
                    continue
                probFour_new[s] = (1 - α) * probFour_new[s] + α * (1 / n_available)
                probKnobby_new[s] = (1 - α) * probKnobby_new[s] + α * (1 / n_available)
                probFour[s] = (1 - α) * probFour[s] + α * (1 / n_available)
                probKnobby[s] = (1 - α) * probKnobby[s] + α * (1 / n_available)

            n = len(all_moves_dict)
            for i in range(len(move_dict)):
                all_moves_dict[n + i] = move_dict[i]
                all_board_dict[n + i] = np.array(board_dict[i])
                all_prob_dict[n + i] = prob_dict[i]
                all_prob_dict_new[n + i] = prob_dict_new[i]
                all_move_pos_first[n + i] = move_pos_first[i]
                all_move_pos_last[n + i] = move_pos_last[i]
                all_rt[n + i] = rt_dict[i]
                all_play_order[n + i] = is_player_first[round]
                all_results[n + i] = games_results[round - 1]

            # NEW save if new prob files are not found
            # -------------------------
            # if not os.path.exists(check_path_k) or not os.path.exists(check_path_f):
            # save_game_data_simple(full_knobby, path_k_new)
            # save_game_data_simple(full_fiar, path_f_new)
            # -------------------------

    # mask the probability matrices with the move matrices
    for m in range(0, len(all_moves_dict)):
        move = all_moves_dict[m]
        mask = move.astype(bool)
        if len(all_prob_dict[m]) != 0:
            all_prob_dict[m] = [all_prob_dict[m][0][mask], all_prob_dict[m][1][mask]]
        if len(all_prob_dict_new[m]) != 0:
            all_prob_dict_new[m] = [all_prob_dict_new[m][0][mask], all_prob_dict_new[m][1][mask]]

    return (all_board_dict, all_prob_dict, all_moves_dict, all_prob_dict_new, all_games_round,
            all_move_pos_first, all_move_pos_last, all_rt, all_play_order, all_results)


def plot_mixed_effect_model(df_blocked, df_interleaved, *axes):
    mode = 3
    # version 1 plot of probability difference
    ax_b_1a = axes[0]
    ax_b_2a = axes[1]
    ax_i_1a = axes[2]
    ax_i_2a = axes[3]

    # version 2 plot of probability difference
    ax_b_1b = axes[4]
    ax_i_1b = axes[5]

    # rt
    ax_b_1c = axes[6]
    ax_i_1c = axes[7]

    if mode != 2:
        # == reaction time
        df_b_first = df_blocked[df_blocked['game_type'] == 0].copy()
        df_b_second = df_blocked[df_blocked['game_type'] == 1].copy()
        df_i_first = df_interleaved[df_interleaved['game_type'] == 0].copy()
        df_i_second = df_interleaved[df_interleaved['game_type'] == 1].copy()
        df_list = [df_b_first, df_b_second, df_i_first, df_i_second]
        for df in df_list:
            if (df is df_b_first) or (df is df_i_first):
                color_light = c_green_1
                color_dark = c_green_2
            else:
                color_light = c_purple_1
                color_dark = c_purple_2

            if (df is df_b_first) or (df is df_b_second):
                ax = ax_b_1c
            else:
                ax = ax_i_1c

            model = smf.mixedlm('rt ~ frac_currGame', df, groups=df['id'], re_formula="1 + frac_currGame")
            result = model.fit(method='nm', maxiter=5000, ftol=1e-2)
            print("summaries in order of blocked first game, blocked second game, "
                  "interleaved first game, and interleaved second game\n")
            print(result.summary())
            df.loc[:, 'fittedvalues'] = result.fittedvalues
            slope = result.params['frac_currGame']
            intercept = result.params['Intercept']
            group_variance = result.cov_re.iloc[0, 0]
            label = f'y = {slope:.3f}x+{intercept:.3f}\n(group variance: {group_variance:.3f})'

            # iterate through each participant to plot individual lines
            for id in df['id'].unique():
                df_id = df[df['id'] == id]
                ax.plot('frac_currGame', 'fittedvalues', data=df_id, color=color_light, alpha=0.4, label='_nolegend_')

            # general trend line
            frac_moves_range = np.linspace(0, 1, 100)
            predicted_rt = intercept + slope * frac_moves_range
            ax.plot(frac_moves_range, predicted_rt, color=color_dark, label=label)

        # Set labels and titles for all subplots
        for ax in [ax_b_1c, ax_i_1c]:
            ax.legend(loc='upper right', fontsize=10)

        ax_b_1c.set_xlabel('Game Progress (Fraction)')
        ax_b_1c.set_ylabel('Reaction time')
        ax_b_1c.set_title('Blocked (n=13)')
        ax_i_1c.set_title('Interleaved (n=13)')

        ax_b_1c.set_xlim(0, 1)
        ax_i_1c.set_xlim(0, 1)
        # set y lim based on quantile q=0.95
        q = df_blocked['rt'].quantile(0.95)
        ax_b_1c.set_ylim(0, q)
        q = df_interleaved['rt'].quantile(0.95)
        ax_i_1c.set_ylim(0, q)

    if mode != 1:
        # v2 === Blocked condition
        model_blocked = smf.mixedlm('prob_diff ~ game_type * frac_currGame', df_blocked, groups=df_blocked['id'],
                                    re_formula="1 + frac_currGame")
        result_blocked_mle = model_blocked.fit(method='nm', maxiter=5000, ftol=1e-2)
        print("result_blocked_mle\n")
        print(result_blocked_mle.summary())
        slope_1 = result_blocked_mle.params['frac_currGame']
        intercept_1 = result_blocked_mle.params['Intercept']
        slope_2 = slope_1 + result_blocked_mle.params['game_type:frac_currGame']
        intercept_2 = intercept_1 + result_blocked_mle.params['game_type']
        group_variance = result_blocked_mle.cov_re.iloc[0, 0]
        label_1 = f'y = {slope_1:.3f}x+{intercept_1:.3f}'
        label_2 = f'y = {slope_2:.3f}x{intercept_2:.3f}\n(group variance: {group_variance:.3f})'
        df_blocked.loc[:, 'fittedvalues'] = result_blocked_mle.fittedvalues
        df_b_first = df_blocked[df_blocked['game_type'] == 0].copy()
        df_b_second = df_blocked[df_blocked['game_type'] == 1].copy()

        # iterate through each participant to plot individual lines
        for id in df_b_first['id'].unique():
            df_id = df_b_first[df_b_first['id'] == id]
            if id == df_b_first['id'].unique()[0]:
                ax_b_1b.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_green_2, alpha=0.4, label=label_1)
            else:
                ax_b_1b.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_green_2, alpha=0.4,
                             label='_nolegend_')
        for id in df_b_second['id'].unique():
            df_id = df_b_second[df_b_second['id'] == id]
            if id == df_b_second['id'].unique()[0]:
                ax_b_1b.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_purple_2, alpha=0.4, label=label_2)
            else:
                ax_b_1b.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_purple_2, alpha=0.4,
                             label='_nolegend_')

        ax_b_1b.scatter('frac_currGame', 'prob_diff', data=df_b_first,
                        color=c_green_1, alpha=0.4, s=5, label='_nolegend_')
        ax_b_1b.scatter('frac_currGame', 'prob_diff', data=df_b_second,
                        color=c_purple_1, alpha=0.4, s=5, label='_nolegend_')

        # v2 === Interleaved condition
        model_interleaved = smf.mixedlm('prob_diff ~ game_type * frac_currGame', df_interleaved,
                                        groups=df_interleaved['id'], re_formula="1 + frac_currGame")
        result_interleaved_mle = model_interleaved.fit(method='nm', maxiter=5000, ftol=1e-2)
        print("result_interleaved_mle\n")
        print(result_interleaved_mle.summary())
        slope_1 = result_interleaved_mle.params['frac_currGame']
        intercept_1 = result_interleaved_mle.params['Intercept']
        slope_2 = slope_1 + result_interleaved_mle.params['game_type:frac_currGame']
        intercept_2 = intercept_1 + result_interleaved_mle.params['game_type']
        group_variance = result_interleaved_mle.cov_re.iloc[0, 0]
        label_1 = f'y = {slope_1:.3f}x+{intercept_1:.3f}'
        label_2 = f'y = {slope_2:.3f}x{intercept_2:.3f}\n(group variance: {group_variance:.3f})'
        df_interleaved.loc[:, 'fittedvalues'] = result_interleaved_mle.fittedvalues

        # iterate through each participant to plot individual lines
        df_i_first = df_interleaved[df_interleaved['game_type'] == 0].copy()
        df_i_second = df_interleaved[df_interleaved['game_type'] == 1].copy()
        for id in df_i_first['id'].unique():
            df_id = df_i_first[df_i_first['id'] == id]
            if id == df_i_first['id'].unique()[0]:
                ax_i_1b.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_green_2, alpha=0.4, label=label_1)
            else:
                ax_i_1b.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_green_2, alpha=0.4,
                             label='_nolegend_')
        for id in df_i_second['id'].unique():
            df_id = df_i_second[df_i_second['id'] == id]
            if id == df_i_second['id'].unique()[0]:
                ax_i_1b.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_purple_2, alpha=0.4, label=label_2)

            else:
                ax_i_1b.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_purple_2, alpha=0.4,
                             label='_nolegend_')

        ax_i_1b.scatter('frac_currGame', 'prob_diff', data=df_i_first,
                        color=c_green_1, alpha=0.4, s=5, label='_nolegend_')
        ax_i_1b.scatter('frac_currGame', 'prob_diff', data=df_i_second,
                        color=c_purple_1, alpha=0.4, s=5, label='_nolegend_')

        # Set labels and titles for all subplots
        for ax in [ax_b_1b, ax_i_1b]:
            ax.legend(loc='upper right')
        ax_b_1b.set_xlabel('Game Progress (Fraction)')
        ax_b_1b.set_ylabel('Probability Difference')
        ax_b_1b.set_title('Blocked (n=13)')
        ax_i_1b.set_title('Interleaved (n=13)')

        ax_b_1b.legend(loc='upper right', fontsize=10)
        ax_i_1b.legend(loc='upper right', fontsize=10)

        ax_b_1b.set_xlim(0, 1)
        ax_i_1b.set_xlim(0, 1)
        ax_b_1b.set_ylim(-4, 4)
        ax_i_1b.set_ylim(-4, 4)

    if mode != 0:
        # v1 === Blocked condition
        df_blocked_first_half = df_blocked[df_blocked['game_type'] == 0].copy()
        df_blocked_second_half = df_blocked[df_blocked['game_type'] == 1].copy()

        # = First half of blocked data
        ax_b_1a.scatter('frac_currGame', 'prob_diff', data=df_blocked_first_half,
                        color=c_green_1, alpha=0.3, s=5, label='_nolegend_')
        model_blocked_first_half = smf.mixedlm('prob_diff ~ frac_currGame', df_blocked_first_half,
                                               groups=df_blocked_first_half['id'], re_formula="1 + frac_currGame")
        result_blocked_mle_1 = model_blocked_first_half.fit(method='nm', maxiter=5000, ftol=1e-2)
        print("result_blocked_mle_1\n")
        print(result_blocked_mle_1.summary())
        slope = result_blocked_mle_1.params['frac_currGame']
        intercept = result_blocked_mle_1.params['Intercept']
        group_variance = result_blocked_mle_1.cov_re.iloc[0, 0]  # Accessing the variance for the random intercept
        label = f'Slope: {slope:.3f}\nIntercept: {intercept:.3f}'
        df_blocked_first_half.loc[:, 'fittedvalues'] = result_blocked_mle_1.fittedvalues
        # ax_b_1a.plot('frac_currGame', 'fittedvalues', data=df_blocked_first_half, color=c_green_2, alpha=0.2, label='_nolegend_')
        # iterate through each participant to plot individual lines
        for id in df_blocked_first_half['id'].unique():
            df_id = df_blocked_first_half[df_blocked_first_half['id'] == id]
            ax_b_1a.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_green_2, alpha=0.4, label='_nolegend_')
        # general trend line
        frac_moves_range = np.linspace(0, 1, 100)
        predicted_prob_diff = intercept + slope * frac_moves_range
        ax_b_1a.plot(frac_moves_range, predicted_prob_diff, color=c_black, label=label)

        # = Second half of blocked data
        ax_b_2a.scatter('frac_currGame', 'prob_diff', data=df_blocked_second_half,
                        color=c_purple_1, alpha=0.3, s=5, label='_nolegend_')
        model_blocked_second_half = smf.mixedlm('prob_diff ~ frac_currGame', df_blocked_second_half,
                                                groups=df_blocked_second_half['id'], re_formula="1 + frac_currGame")
        result_blocked_mle_2 = model_blocked_second_half.fit(method='nm', maxiter=5000, ftol=1e-2)
        print("result_blocked_mle_2\n")
        print(result_blocked_mle_2.summary())
        slope = result_blocked_mle_2.params['frac_currGame']
        intercept = result_blocked_mle_2.params['Intercept']
        group_variance = result_blocked_mle_2.cov_re.iloc[0, 0]
        label = f'Slope: {slope:.3f}\nIntercept: {intercept:.3f}'
        df_blocked_second_half.loc[:, 'fittedvalues'] = result_blocked_mle_2.fittedvalues
        # ax_b_2a.plot('frac_currGame', 'fittedvalues', data=df_blocked_second_half, color=c_purple_2, alpha=0.2, label='_nolegend_')
        # iterate through each participant to plot individual lines
        for id in df_blocked_second_half['id'].unique():
            df_id = df_blocked_second_half[df_blocked_second_half['id'] == id]
            ax_b_2a.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_purple_2, alpha=0.4, label='_nolegend_')

        # general trend line
        frac_moves_range = np.linspace(0, 1, 100)
        predicted_prob_diff = intercept + slope * frac_moves_range
        ax_b_2a.plot(frac_moves_range, predicted_prob_diff, color=c_black, label=label)

        # v1 === Interleaved condition
        df_interleaved_odd = df_interleaved[df_interleaved['game_type'] == 0].copy()
        df_interleaved_even = df_interleaved[df_interleaved['game_type'] == 1].copy()

        # = Odd indices of interleaved data
        ax_i_1a.scatter('frac_currGame', 'prob_diff', data=df_interleaved_odd,
                        color=c_green_1, alpha=0.3, s=5, label='_nolegend_')
        model_interleaved_odd = smf.mixedlm('prob_diff ~ frac_currGame', df_interleaved_odd,
                                            groups=df_interleaved_odd['id'], re_formula="1 + frac_currGame")
        result_interleaved_mle_1 = model_interleaved_odd.fit(method='nm', maxiter=5000, ftol=1e-2)
        print("result_interleaved_mle_1\n")
        print(result_interleaved_mle_1.summary())
        slope = result_interleaved_mle_1.params['frac_currGame']
        intercept = result_interleaved_mle_1.params['Intercept']
        group_variance = result_interleaved_mle_1.cov_re.iloc[0, 0]  # Accessing the variance for the random intercept
        label = f'Slope: {slope:.3f}\nIntercept: {intercept:.3f}'
        df_interleaved_odd['fittedvalues'] = result_interleaved_mle_1.fittedvalues
        # ax_i_1a.plot('frac_currGame', 'fittedvalues', data=df_interleaved_odd, color=c_green_2, alpha=0.2, label='_nolegend_')
        # iterate through each participant to plot individual lines
        for id in df_interleaved_odd['id'].unique():
            df_id = df_interleaved_odd[df_interleaved_odd['id'] == id]
            ax_i_1a.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_green_2, alpha=0.4, label='_nolegend_')

        # General trend line
        frac_moves_range = np.linspace(0, 1, 100)
        predicted_prob_diff = intercept + slope * frac_moves_range
        ax_i_1a.plot(frac_moves_range, predicted_prob_diff, color=c_black, label=label)

        # = Even indices of interleaved data
        ax_i_2a.scatter('frac_currGame', 'prob_diff', data=df_interleaved_even,
                        color=c_purple_1, alpha=0.3, s=5, label='_nolegend_')
        model_interleaved_even = smf.mixedlm('prob_diff ~ frac_currGame', df_interleaved_even,
                                             groups=df_interleaved_even['id'], re_formula="1 + frac_currGame")
        result_interleaved_mle_2 = model_interleaved_even.fit(method='nm', maxiter=5000, ftol=1e-2)
        print("result_interleaved_mle_2\n")
        print(result_interleaved_mle_2.summary())
        slope = result_interleaved_mle_2.params['frac_currGame']
        intercept = result_interleaved_mle_2.params['Intercept']
        group_variance = result_interleaved_mle_2.cov_re.iloc[0, 0]  # Accessing the variance for the random intercept
        label = f'Slope: {slope:.3f}\nIntercept: {intercept:.3f}'
        df_interleaved_even['fittedvalues'] = result_interleaved_mle_2.fittedvalues
        # ax_i_2a.plot('frac_currGame', 'fittedvalues', data=df_interleaved_even, color=c_purple_2, alpha=0.2, label='_nolegend_')
        # iterate through each participant to plot individual lines
        for id in df_interleaved_even['id'].unique():
            df_id = df_interleaved_even[df_interleaved_even['id'] == id]
            ax_i_2a.plot('frac_currGame', 'fittedvalues', data=df_id, color=c_purple_2, alpha=0.4, label='_nolegend_')

        # General trend line
        frac_moves_range = np.linspace(0, 1, 100)
        predicted_prob_diff = intercept + slope * frac_moves_range
        ax_i_2a.plot(frac_moves_range, predicted_prob_diff, color=c_black, label=label)

        # Set labels and titles for all subplots
        ax_i_1a.set_ylabel('Probability Difference')
        ax_i_1a.set_xlabel('Game Progress (Fraction)')
        for ax in [ax_b_1a, ax_b_2a, ax_i_1a, ax_i_2a]:
            ax.legend(loc='upper right')
            ax.set_xlim(0, 1)

        ax_b_1a.set_title('Blocked, First Game (n=13)')
        ax_b_2a.set_title('Blocked, Second Game (n=13)')
        ax_i_1a.set_title('Interleaved, First Game (n=13)')
        ax_i_2a.set_title('Interleaved, Second Game (n=13)')


def plot_move_prob_comparison(ax1, ax2, dataFrame, condition):
    y_diff = []
    all_round = []
    all_move = []
    all_move_fraction = []
    all_rt = []
    # max_moves = dataFrame.groupby('id').size().max()

    for i in range(len(dataFrame)):
        all_round.append(dataFrame['round'].iloc[i])
        all_move.append(dataFrame['move_number'].iloc[i])
        all_move_fraction.append(dataFrame['frac_moves'].iloc[i])
        all_rt.append(dataFrame['rt'].iloc[i])
        # y_diff.append(dataFrame['prob_diff_new'].iloc[i])
        y_diff.append(dataFrame['prob_diff'].iloc[i])

    x_fraction = np.array(all_move_fraction)
    y_diff = [0 if v is None else v for v in y_diff]
    y_diff = np.array(y_diff, dtype=float)

    x_1 = []
    y_1 = []
    x_2 = []
    y_2 = []
    rt_1 = []
    rt_2 = []

    half_fraction = 0.5
    if (condition == 0):  # blocked learning
        for i in all_move_fraction:
            index = all_move_fraction.index(i)
            if i < half_fraction:
                x_1.append(i)
                y_1.append(y_diff[index])
                rt_1.append(all_rt[index])
            if i >= half_fraction:
                x_2.append(i)
                y_2.append(y_diff[index])
                rt_2.append(all_rt[index])

    elif (condition == 1):  # interleaved learning
        # plot odd and even data separately
        for i, round_num in enumerate(all_round):  # i = index, move_num = move number
            if (round_num % 2 != 0):  # first game
                x_1.append(x_fraction[i])
                y_1.append(y_diff[i])
                rt_1.append(all_rt[i])
            else:
                x_2.append(x_fraction[i])
                y_2.append(y_diff[i])
                rt_2.append(all_rt[i])

    y_1 = np.array(y_1)
    y_2 = np.array(y_2)
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    rt_1 = np.array(rt_1)
    rt_2 = np.array(rt_2)

    model_1 = np.poly1d(np.polyfit(x_1, y_1, 2))
    y_smooth_1 = model_1(x_1)
    model_2 = np.poly1d(np.polyfit(x_2, y_2, 2))
    y_smooth_2 = model_2(x_2)
    model_rt_1 = np.poly1d(np.polyfit(x_1, rt_1, 2))
    rt_smooth_1 = model_rt_1(x_1)
    model_rt_2 = np.poly1d(np.polyfit(x_2, rt_2, 2))
    rt_smooth_2 = model_rt_2(x_2)

    # Formatting coefficients for the label
    # label_1 = f'y = {coef_1[0]:.2f}x^3 + ...\n{coef_1[3]:.2f}'
    # label_2 = f'y = {coef_2[0]:.2f}x^3 + ...\n{coef_2[3]:.2f}'

    ax1.plot(x_1, y_1, 'o', color=c_orange_1, markersize=2, alpha=0.3)
    ax2.plot(x_2, y_2, 'o', color=c_orange_1, markersize=2, alpha=0.3)
    ax1.plot(x_1, y_smooth_1, color=c_orange_2, label='_nolegend_')
    ax2.plot(x_2, y_smooth_2, color=c_orange_2, label='_nolegend_')
    ax1_rt = ax1.twinx()
    ax2_rt = ax2.twinx()

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    rt_combined = np.concatenate((rt_1, rt_2))

    # Calculate the 25% and 75% quantiles for the y-axis limits
    y_max_quantile = np.percentile(rt_combined, 75)  # Upper limit based on 75% quantile

    # Round the quantile values to the nearest integers and expand to avoid edge-aligned ticks
    y_min_expanded = 0
    y_max_expanded = int(np.ceil(y_max_quantile)) + 1

    step_size = max((y_max_expanded - y_min_expanded) // 6, 1)
    ticks = np.arange(y_min_expanded, y_max_expanded + 1, step_size)

    # Extend the y-axis limits beyond the min and max tick values to avoid edge alignment
    y_lim_lower = ticks[0] - step_size / 2 if len(ticks) > 0 else y_min_expanded
    y_lim_upper = ticks[-1] + step_size / 2 if len(ticks) > 0 else y_max_expanded

    # Apply the same tick locations to both ax1_rt and ax2_rt
    ax1_rt.set_yticks(ticks)
    ax2_rt.set_yticks(ticks)

    # Set the y-axis limits with a buffer
    ax1_rt.set_ylim(y_lim_lower, y_lim_upper)
    ax2_rt.set_ylim(y_lim_lower, y_lim_upper)

    ax1_rt.plot(x_1, rt_smooth_1, color=c_blue_2, label='RT')
    ax2_rt.plot(x_2, rt_smooth_2, color=c_blue_2, label='RT')

    ax1.tick_params(axis='y', labelcolor=c_orange_2)
    ax1_rt.tick_params(axis='y', labelcolor=c_blue_2)
    ax2.tick_params(axis='y', labelcolor=c_orange_2)
    ax2_rt.tick_params(labelright=False)

    if (condition == 0):
        ax1.set_title('Blocked Condition (first game)', fontsize=10)
        ax2.set_title('Blocked Condition (second game)', fontsize=10)
        ax1.set_xlabel('Move Fraction (0 - 1)')
        ax1.set_xlim(0, half_fraction)
        ax2.set_xlim(half_fraction, 1)
    elif (condition == 1):
        ax1.set_title('Interleaved Condition (first game)', fontsize=10)
        ax2.set_title('Interleaved Condition (second game)', fontsize=10)
        ax1.set_xlabel('Move Fraction (0 - 1)')
        ax1.set_xlim(0, 1)
        ax2.set_xlim(0, 1)

    ax1.legend(loc='upper right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax1_rt.legend(loc='upper left', fontsize=8)
    ax2_rt.legend(loc='upper left', fontsize=8)


def normalize_and_diff_prob(prob, first_game):
    # a new list with length of prob
    diff = [0] * len(prob)
    base = 10

    # log 10
    for i in range(len(prob)):
        prob_fiar = prob[i][0]
        prob_knobby = prob[i][1]

        if (prob_fiar != 0):
            prob_fiar = math.log(prob_fiar[0], base)
        else:
            prob_fiar = 0

        if (prob_knobby != 0):
            prob_knobby = math.log(prob_knobby[0], base)
        else:
            prob_knobby = 0

        if (first_game == 0):  # fiar
            diff[i] = prob_fiar - prob_knobby
        elif (first_game == 1):  # knobby
            diff[i] = prob_knobby - prob_fiar

        diff[i] = float(diff[i])

    return diff


def remove_outliers(data):
    # remove outliers
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    threshold = 1.5

    # Identify outliers
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    # outliers = (data < lower_bound) | (data > upper_bound)
    # data_filtered = [0 if d in outliers else d for d in data]
    # return data_filtered

    # ALTERNATIVE: Use a boolean mask for indices within the interquartile range
    inliers = (data >= lower_bound) & (data <= upper_bound)

    return inliers


def check_data_quality(all_data):
    # a dictionary to store the win rate for each participant
    win_rate_dict = {}

    for file_name, file_paths in all_data.items():
        for path in file_paths:
            with open(path, 'r') as file:
                data = json.load(file)
            if (data['participant_id'] not in deprecated_id):
                # calculate individual win rate
                results = data.get('games_results', [])
                win_result = [0 if r != 1 else 1 for r in results]
                game_total = len(results)
                win_rate = round(sum(win_result) / game_total, 3)

                # print("id", data['participant_id'], "win_rate", win_rate, "for game_total", game_total)

                win_rate_dict[data['participant_id']] = win_rate

    # plot win rate distribution
    n_bin = round(math.sqrt(len(win_rate_dict)))
    plt.figure(figsize=(4, 4), dpi=300)
    plt.hist(win_rate_dict.values(), bins=5, alpha=0.8, color=c_black, edgecolor='white')
    plt.xlabel('Win Rate')
    plt.ylabel('Frequency')
    plt.yticks(range(0, 10, 1))
    plt.title('Win Rate Distribution (n={})'.format(len(win_rate_dict)))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.show()


def create_dataframe(id_blocked, id_interleaved):
    data = []
    for id in id_blocked + id_interleaved:
        board, prob, move, prob_new, round, is_first_move, is_last_move, rt, play_order, result = get_all_move_prob(id)

        prob_four = []
        prob_knobby = []
        prob_new_four = []
        prob_new_knobby = []
        for i in range(len(prob)):
            prob_four.append(prob[i][0])
            prob_knobby.append(prob[i][1])
        for i in range(len(prob_new)):
            prob_new_four.append(prob_new[i][0])
            prob_new_knobby.append(prob_new[i][1])

        first_game, game_type_list = get_game_sequence_by_id(id)
        game_type = []
        for i in range(len(game_type_list)):
            for j in range(len(round)):
                if (round[j] == i + 1):
                    game_type.append(game_type_list[i])

        # count rounds for each game type
        count_first_game = 0
        count_second_game = 0
        for i in range(len(game_type)):
            if game_type[i] == first_game:
                count_first_game += 1
            else:
                count_second_game += 1

        prob = normalize_and_diff_prob(prob, first_game)
        prob_new = normalize_and_diff_prob(prob_new, first_game)
        condition = 'blocked' if id in id_blocked else 'interleaved'
        first_game = 'fiar' if first_game == 0 else 'knobby'
        # values = np.array(list(rt.values()))
        # inlier_mask = remove_outliers(values)
        # filtered_rt = {key: (value if mask else None) for key, value, mask in zip(rt.keys(), rt.values(), inlier_mask)}

        currGame1 = 0
        currGame2 = 0
        for i in range(len(prob)):
            frac_moves = (i + 1) / len(prob)

            if game_type[i] == 0:
                currGame1 += 1
                frac_currGame = currGame1 / count_first_game
            else:
                currGame2 += 1
                frac_currGame = currGame2 / count_second_game

            data.append([id, condition, first_game, round[i], game_type[i],
                         i + 1, frac_moves, frac_currGame, board[i],
                         prob[i], prob_new[i], prob_four[i], prob_knobby[i], prob_new_four[i], prob_new_knobby[i],
                         rt[i],
                         is_first_move[i], is_last_move[i], play_order[i], result[i]])
    # df = None
    df = pd.DataFrame(data,
                      columns=['id', 'condition', 'first_game', 'round', 'game_type',
                               'move_number', 'frac_moves', 'frac_currGame', 'context',
                               'prob_diff', 'prob_diff_new', 'four_old', 'knobby_old', 'four_new', 'knobby_new',
                               'rt',
                               'is_first_move', 'is_last_move', 'play_order', 'result'])

    # test =================
    # Calculate descriptive statistics for prob_diff
    # stats_prob_diff = df['prob_diff'].describe()
    #
    # # Calculate descriptive statistics for prob_diff_new
    # stats_prob_diff_new = df['prob_diff_new'].describe()
    #
    # # Print the descriptive statistics
    # print("Descriptive Statistics for prob_diff:")
    # print(stats_prob_diff)
    # print("\nDescriptive Statistics for prob_diff_new:")
    # print(stats_prob_diff_new)
    # plt.figure(figsize=(12, 6))
    #
    # # Histogram for prob_diff
    # plt.subplot(1, 2, 1)
    # plt.hist(df['prob_diff'].dropna(), bins=30, color='skyblue', edgecolor='black')
    # plt.title('Histogram of prob_diff')
    # plt.xlabel('prob_diff')
    # plt.ylabel('Frequency')
    #
    # # Histogram for prob_diff_new
    # plt.subplot(1, 2, 2)
    # plt.hist(df['prob_diff_new'].dropna(), bins=30, color='orange', edgecolor='black')
    # plt.title('Histogram of prob_diff_new')
    # plt.xlabel('prob_diff_new')
    # plt.ylabel('Frequency')
    #
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(3, 6))
    #
    # # Box plot for prob_diff and prob_diff_new
    # # plt.boxplot([df['prob_diff'].dropna(), df['prob_diff_new'].dropna()], labels=['prob_diff', 'prob_diff_new'])
    # plt.boxplot(df['prob_diff'].dropna(), labels=['prob_diff'])
    # plt.title('Data Distribution')
    # plt.ylabel('Values')
    #
    # plt.show()

    # ======================
    return df


def main():
    global params_list, \
        paths_blocked, \
        paths_interleaved, \
        sum_blocked, \
        sum_interleaved, \
        win_rate_blocked_1, \
        win_rate_interleaved_1, \
        win_rate_blocked_2, \
        win_rate_interleaved_2, \
        win_rate_knobby_all, \
        win_rate_fiar_all, \
        id_blocked, \
        id_interleaved

    # Find params in each folder
    params_list = find_duplicate_params()

    # test overall data quality
    check_data_quality(params_list)

    # Group data by condition (blocked vs interleaved)
    group_by_condition(params_list)

    # people's first game results and second game results
    data_blocked_1 = defaultdict(list)
    data_interleaved_1 = defaultdict(list)
    data_blocked_2 = defaultdict(list)
    data_interleaved_2 = defaultdict(list)
    results_knobby_all = defaultdict(list)
    results_fiar_all = defaultdict(list)


    # how many people played a specific game first in a condition
    count_blocked_first_4iar = 0
    count_blocked_first_knobby = 0
    count_mix_first_4iar = 0
    count_mix_first_knobby = 0


    # 1) compare first & second game results

    for params_path in paths_blocked:
        id, first_game, results_knobby, results_four = calculate_win(params_path)
        results_knobby_all[id] = results_knobby
        results_fiar_all[id] = results_four

        if (first_game == 0):
            data_blocked_1[id] = results_four
            data_blocked_2[id] = results_knobby
            count_blocked_first_4iar += 1
        elif (first_game == 1):
            data_blocked_1[id] = results_knobby
            data_blocked_2[id] = results_four
            count_blocked_first_knobby += 1

    for params_path in paths_interleaved:
        id, first_game, results_knobby, results_four = calculate_win(params_path)
        results_knobby_all[id] = results_knobby
        results_fiar_all[id] = results_four

        if (first_game == 0):
            data_interleaved_1[id] = results_four
            data_interleaved_2[id] = results_knobby
            count_mix_first_4iar += 1
        elif (first_game == 1):
            data_interleaved_1[id] = results_knobby
            data_interleaved_2[id] = results_four
            count_mix_first_knobby += 1

    win_rate_blocked_1, max_b_1 = get_win_rate_all(data_blocked_1)
    win_rate_interleaved_1, max_i_1 = get_win_rate_all(data_interleaved_1)
    win_rate_blocked_2, max_b_2 = get_win_rate_all(data_blocked_2)
    win_rate_interleaved_2, max_i_2 = get_win_rate_all(data_interleaved_2)
    win_rate_knobby_all, max_k = get_win_rate_all(results_knobby_all)
    win_rate_fiar_all, max_f = get_win_rate_all(results_fiar_all)

    # create dataframe for plotting
    df = create_dataframe(id_blocked, id_interleaved)

    count = list(zip([count_blocked_first_4iar,
                      count_blocked_first_knobby,
                      count_mix_first_4iar,
                      count_mix_first_knobby]))
    plot_win_rate(count, df)

    # 2) plot individual move probability comparison
    # blocked

    # for id in id_interleaved:
    # for id in id_blocked:
    #     fig_comparison, (ax_b_1, ax_b_2) = plt.subplots(1, 2, figsize=(6, 4), sharey=True)
    #     dataFrame = df[df['id'] == id]
    #     condition = 0 if dataFrame['condition'].iloc[0] == 'blocked' else 1
    #     plot_move_prob_comparison(ax_b_1, ax_b_2, dataFrame, condition)
    #     fig_comparison.suptitle(f'Participant {id}', fontsize=10)
    #     fig_comparison.show()

    # 3) mixed effect model
    df_blocked_copy = df[df['condition'] == 'blocked'].copy()
    df_interleaved_copy = df[df['condition'] == 'interleaved'].copy()

    # access the id column
    df_blocked_copy.loc[:, 'id'] = df_blocked_copy['id'].astype(str)
    df_blocked_copy.loc[:, 'id'] = pd.Categorical(df_blocked_copy['id'])
    df_interleaved_copy.loc[:, 'id'] = df_interleaved_copy['id'].astype(str)
    df_interleaved_copy.loc[:, 'id'] = pd.Categorical(df_interleaved_copy['id'])

    fig_mle_1, ((ax_blocked_1a, ax_blocked_2a), (ax_interleaved_1a, ax_interleaved_2a)) = plt.subplots(2, 2,
                                                                                                       figsize=(10, 8),
                                                                                                       sharey=True,
                                                                                                       dpi=300)
    fig_mle_2, (ax_blocked_b, ax_interleaved_b) = plt.subplots(1, 2, figsize=(8, 4), sharey=True, dpi=300)
    plt.tight_layout(pad=3.0)
    fig_mle_rt, (ax_blocked_rt, ax_interleaved_rt) = plt.subplots(1, 2, figsize=(8, 4), sharey=True, dpi=300)
    plt.tight_layout(pad=3.0)
    plot_mixed_effect_model(df_blocked_copy, df_interleaved_copy,
                            ax_blocked_1a, ax_blocked_2a, ax_interleaved_1a, ax_interleaved_2a,
                            ax_blocked_b, ax_interleaved_b,
                            ax_blocked_rt, ax_interleaved_rt)

    plt.show()


if __name__ == "__main__":
    main()
