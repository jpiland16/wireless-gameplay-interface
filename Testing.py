print("\nLoading...\n")

from math import log2
import pickle
from copy import deepcopy

from neural_networks.base.NetworkInteraction import get_adversaries, \
    get_policy_makers, get_receivers, get_transmitters, play_games, \
    get_game_params_from_dict
from neural_networks.base.ParameterHost import get_parameters

policy_maker_list = get_policy_makers()
transmitter_list = get_transmitters()
receiver_list = get_receivers()
adversary_list = get_adversaries()

# Choose the players
policy_maker = policy_maker_list[1]
transmitter = transmitter_list[3]
receiver = receiver_list[0]
adversary = adversary_list[1]

def main():
    # Number of repeats for each simulation
    repeats = 250

    # Print info about what we are doing
    print(f"Running {repeats} sims. ea. for P = {str(policy_maker)}, \n" + 
        f"T = {str(transmitter)}, R = {str(receiver)}, A = {str(adversary)}\n")

    default_params = get_parameters("GAME_PARAMS")

    """ 
    Score storage format
    [
    num_bands 10 [ num_policies 10 ... 100]
    .
    .
    .
    num_bands 100 [ num_policies 10 ... 100]
    ]
    """
    b_scores = []

    count = 1

    for num_bands in range(10, 100):
        b_scores.append([])

        for num_policies in range(10, 100, 10):

            print(f"\nRun {count} of 810...")

            new_params = deepcopy(default_params)
            new_params["M"] = num_bands
            new_params["N"] = num_policies

            completed_games = play(
                policy_maker=policy_maker,
                transmitter=transmitter,
                receiver=receiver,
                adversary=adversary,
                count=repeats,
                params=new_params
            )

            b_avg_score = sum([game.state.score_b for game in completed_games]) / \
                len(completed_games)

            b_scores[-1].append(b_avg_score)

            count += 1

    pickle.dump(b_scores, open("all-data.pkl", 'wb'))

def play(policy_maker, transmitter, receiver, adversary, count, params): 
    return play_games(
        train_model=False,
        print_each_game=False,
        nnet_params=None,
        game_params=get_game_params_from_dict(params),
        policy_maker=policy_maker,
        transmitter=transmitter,
        receiver=receiver,
        adversary=adversary,
        count=count,
        show_output=True
    )

class SimResult:
    def __init__(self, games, params):
        self.games = games
        self.params = params

def main2(adversary, name):

    # Number of repeats for each simulation
    repeats = 2

    policy_maker = policy_maker_list[1]
    transmitter = transmitter_list[3]
    receiver = receiver_list[0]

    # Print info about what we are doing
    print(f"Running {repeats} sims. ea. for P = {str(policy_maker)}, \n" + 
        f"T = {str(transmitter)}, R = {str(receiver)}, A = {str(adversary)}\n")

    default_params = get_parameters("GAME_PARAMS")

    range_bands = range(10, 110, 10)
    range_policies = range(10, 110, 10)
    total_count = len(range_bands) * len(range_policies)

    results = [ ]

    count = 1

    for num_bands in range_bands:

        for num_policies in range_policies:

            print(f"\nRun {count} of {total_count}...")

            new_params = deepcopy(default_params)
            new_params["M"] = num_bands
            new_params["N"] = num_policies
            new_params["R1"] = log2(num_policies) + 3

            completed_games = play(
                policy_maker=policy_maker,
                transmitter=transmitter,
                receiver=receiver,
                adversary=adversary,
                count=repeats,
                params=new_params
            )

            result = SimResult(completed_games, new_params)

            results.append(result)

            count += 1

    pickle.dump(results, open(name, 'wb'))

if __name__ == "__main__":
    main2(adversary_list[1], "gamma.pkl")
    main2(adversary_list[3], "rlrnn.pkl")