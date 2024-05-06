"""


@author: Hannah Zeng
"""
# -*- coding: utf-8 -*-

from __future__ import print_function
import subprocess
import os
import numpy as np
import json
import shutil
import sys

NUM_INPUTS = 4

path = ""
path_Data = "Data"
subprocesses = []

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

def main():
    global params, path
    params = {
        "participant_id": "",
        "condition": 0,  # 0: block, 1: interchange
        "model": 0,  # 0: four-in-a-row, 1: knobby
        "state": 0,  # 0: init, 1: in_trial, 2: idle, 3: end
        "trials_fouriar": 0,
        "trials_knobby": 0,
        "max_moves": 0,
        "moves_fouriar": 10,
        "moves_knobby": 10,
        "fouriar_complete": False,
        "knobby_complete": False,
        "games_count": 0,
        "games_rule": [],
        "games_results": [],
        "win_rate_fouriar": 0,
        "win_rate_knobby": 0
    }
    inputs = input("Set up the experiment (space as deliminator): "
                   "\n- participant id (pXX)"
                   "\n- condition (0: block learning, 1: interchange learning) "
                   "\n- rule to start with (0: four-in-a-row, 1: knobby):"
                   "\n- moves to control for:\n")
    inputs = inputs.split(" ")
    if len(inputs) != NUM_INPUTS:
        print("The experiment needs ", NUM_INPUTS, "parameters to set up. Please try again.")
        return

    for idx, param in enumerate(inputs, 0):
        params["participant_id"] = inputs[0]
        params["condition"] = int(inputs[1])
        params["model"] = int(inputs[2])
        params["moves_fouriar"] = int(inputs[3])
        params["moves_knobby"] = int(inputs[3])
        params["max_moves"] = int(inputs[3])

    # print("Participant ID: ", params["participant_id"])

    # if params["condition"] == 0:
    #     print("Condition is learning by block")
    # else:
    #     print("Condition is learning by interchange")

    # if params["model"] == 0:
    #     print("Rule to start with is four-in-a-row")
    # else:
    #     print("Rule to start with is knobby")

    # print("Current moves left:", "4iar =", params["moves_fouriar"], "knobby =", params["moves_knobby"])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

    store_params_to_file()

    update_with_condition(True)


def update_with_condition(is_first=False):
    global params
    params = load_params_from_file()

    evaluate_game()

    if (params["fouriar_complete"] and params["knobby_complete"]):
        print("All trials completed")
        end_experiment()
        return

    # block condition
    if params["condition"] == 0:
        if params["model"] == 0 and params["fouriar_complete"]:
            params["model"] = 1
        elif params["model"] == 1 and params["knobby_complete"]:
            params["model"] = 0

    # interchange condition
    elif params["condition"] == 1:
        if is_first:
            pass
        else:
            if params["model"] == 1:
                params["model"] = 0
            elif params["model"] == 0:
                params["model"] = 1

    store_params_to_file()

    start_a_game()


def start_a_game():
    global params
    params = load_params_from_file()

    # init
    if params["state"] == 0:
        # print("Current state: init")
        evaluate_game()
        params["state"] = 1
        params["games_count"] += 1
        store_params_to_file()

        call_human_play()

    # in_trial
    elif params["state"] == 1:
        # print("Current state: in_trial")
        params["state"] = 2
        store_params_to_file()
        start_a_game()

    # idle (between-trials)
    elif params["state"] == 2:
        # print("Current state: idle")
        if params["moves_fouriar"] <= 0 and params["moves_knobby"] <= 0:
            print()
            print("All trials completed")
            print()
            params["state"] = 3
            store_params_to_file()
            start_a_game() # call self to end experiment

        else:
            print()
            print("- - - Starting another game - - -")
            print()
            evaluate_game()
            params["state"] = 1
            params["games_count"] += 1
            store_params_to_file()

            call_human_play()

    elif params["state"] == 3:
        # print("Current state: end")
        store_params_to_file()
        end_experiment()

        return

def end_experiment():
    params = load_params_from_file()
    move_files_with_id(params["participant_id"])
    print(TextColor.CYAN + "\nExperiment is complete. Thank you for participating!")
    print()
    print()
    print()
    print()

    summary(params)

    # Terminate all subprocesses
    for sp in subprocesses:
        try:
            sp.terminate()  # Send SIGTERM
            sp.wait()  # Wait for the subprocess to exit
        except OSError:
            pass  # Ignore the error if subprocess is already terminated

    # Exit the script
    sys.exit(0)

def evaluate_game():
    global params
    params = load_params_from_file()

    if (params["moves_fouriar"] <= 0):
        params["fouriar_complete"] = True
    if (params["moves_knobby"] <= 0):
        params["knobby_complete"] = True

    store_params_to_file()
    params = load_params_from_file()

    # print("Max moves:", "4iar =", params["moves_fouriar"], "knobby =", params["moves_knobby"])


def call_human_play():
    if params["model"] == 0:
        print()
        print()
        print()
        confirmation = input(
            "Current game rule is " + TextColor.CYAN + "four in a row." + TextColor.RESET + " Type 1 and return to continue.")
        if confirmation == "1":
            params["trials_fouriar"] += 1
        else:
            print("Invalid input. Please try again.")
            call_human_play()


    else:
        print()
        print()
        print()
        confirmation = input(
            "Current game rule is " + TextColor.CYAN + "knobby." + TextColor.RESET + " Type 1 and return to continue.")
        if confirmation == "1":
            params["trials_knobby"] += 1
        else:
            print("Invalid input. Please try again.")
            call_human_play()

    s = subprocess.call(['python', 'human_play.py'])
    subprocesses.append(s)

# helper function

def summary(params):
    # Initialize counters for wins in both game types
    wins_fouriar = 0
    wins_knobby = 0
    games_fouriar = 0
    games_knobby = 0

    # Iterate over the games and count wins and game occurrences for each type
    for i, rule in enumerate(params['games_rule']):
        result = params['games_results'][i]

        if rule == 0:  # Rule 0 is for four-in-a-row games
            games_fouriar += 1
            if result == 1:  # Result 1 is a win for the human player
                wins_fouriar += 1
        elif rule == 1:  # Rule 1 is for knobby games
            games_knobby += 1
            if result == 1:  # Result 1 is a win for the human player
                wins_knobby += 1

    # Calculate win rates for each game type
    win_rate_fouriar = (wins_fouriar / games_fouriar) * 100 if games_fouriar > 0 else 0
    win_rate_knobby = (wins_knobby / games_knobby) * 100 if games_knobby > 0 else 0
    params["win_rate_fouriar"] = win_rate_fouriar
    params["win_rate_knobby"] = win_rate_knobby

    store_params_to_file()


def load_params_from_file(filename="params.json"):
    with open(filename, 'r') as file:
        return json.load(file)

def store_params_to_file(filename="params.json"):
    with open(filename, 'w') as file:
        json.dump(params, file)

def move_files_with_id(participant_id):
    destination_dir = f"Data/{participant_id}/"
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Move each file matching the pattern to the destination directory
    file_path = os.path.join(os.getcwd(), "Data")
    files = os.listdir(file_path)
    for filename in files:
        #print(filename)
        if filename.startswith(f"{participant_id}_"):
            #print(filename, "is moved")
            f = os.path.join(file_path, filename)
            shutil.move(f, destination_dir)

    # Also move the params.json file
    if os.path.exists("Data/p34/params.json"):
        shutil.move("Data/p34/params.json", destination_dir)

def board_to_matrix(self):
    """Converts current board state to a matrix representation."""
    matrix = np.zeros((self.width, self.height), dtype=int)
    for move, player in self.states.items():
        h, w = self. move_to_location(move)
        matrix[h][w] = player
    return matrix


def next_filename(base="data"):
    """Generate the next filename based on existing files."""
    global params
    params = load_params_from_file()

    append_id = str(params["participant_id"])
    games_count = str(params["games_count"])
    abs_dir = os.path.dirname(os.path.abspath(__file__)) + "/Data/"
    dir = os.path.join(abs_dir, f"{append_id}_{base}_{games_count}")

    return os.path.join(dir)


def save_game_data(data, typename="data", threeD=True):
    ext1 = ".npy"
    ext2 = ".txt"
    filename = next_filename(typename) # experiment use
    filename1 = os.path.join(f"{filename}{ext1}")
    filename2 = os.path.join(f"{filename}{ext2}")
    np.save(filename1, data)
    if threeD:
        with open(filename2, 'w') as outfile:
            data_converted = np.array(data, dtype=np.float64)
            for i, slice_ in enumerate(data_converted):
                np.savetxt(outfile, slice_, fmt='%.9e')
                outfile.write('\n')  # Separator line for readability
    else:
        with open(filename2, "w") as file:
            file.write(str(data))

def save_game_data_simple(data, filename, threeD=True):
    ext1 = ".npy"
    ext2 = ".txt"
    filename1 = os.path.join(f"{filename}{ext1}")
    filename2 = os.path.join(f"{filename}{ext2}")
    np.save(filename1, data)
    if threeD:
        with open(filename2, 'w') as outfile:
            data_converted = np.array(data, dtype=np.float64)
            for i, slice_ in enumerate(data_converted):
                np.savetxt(outfile, slice_, fmt='%.9e')
                outfile.write('\n')  # Separator line for readability
    else:
        with open(filename2, "w") as file:
            file.write(str(data))
    print("Data saved to ", filename1, " and ", filename2)

if __name__ == "__main__":
    main()
