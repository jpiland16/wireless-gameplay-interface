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
    def __init__(self, games, params, similarity):
        self.games = games
        self.params = params
        self.similarity = similarity

def jonathan_test_1():

    repeats = 20
    
    all_possible_pairs = []

    #  -- Adversaries
    #  0 - ExampleAdversary
    #  1 - GammaAdversary
    #  3 - PriyaRL_NoPolicy
    #  4 - PriyaRL_WithPolicy

    usable_adversaries = [adversary_list[i] for i in [0, 1, 3, 4]]

    # -- Transmitters
    # 3 - IntelligentTransmitter
    # 4 - PriyaRLTransmitter
    # 5 - RandomTransmitter

    usable_transmitters = [transmitter_list[i] for i in [3, 4, 5]]

    for adversary in usable_adversaries:
        for transmitter in usable_transmitters:
            all_possible_pairs.append((adversary, transmitter))

    policy_maker = policy_maker_list[2]
    receiver = receiver_list[0]

    default_params = get_parameters("GAME_PARAMS")
    params = get_game_params_from_dict(default_params)

    similarities = [i/10 for i in range(10)]

    results = []

    filename = "similarity-test.pkl"

    try:

        for index, (adversary, transmitter) in enumerate(all_possible_pairs):

            for index2, similarity in enumerate(similarities):

                print(f"\n{transmitter.agent.name} VS {adversary.agent.name} " + 
                    f"(Pairing {index + 1} of {len(all_possible_pairs)}) - " +
                    f"Similarity value {index2 + 1} of {len(similarities)}" + 
                    f" = {similarity}\n")

                games = play_games(
                    train_model=False,
                    print_each_game=False,
                    nnet_params=None,
                    game_params=params,
                    policy_maker=policy_maker,
                    transmitter=transmitter,
                    receiver=receiver,
                    adversary=adversary,
                    count=repeats,
                    show_output=True,
                    pm_sim_score = similarity
                )

                result = SimResult(games, params, similarity)

                results.append(result)

    except:
        save_results(filename, results)
        
        print("ABORTING TRIALS - FATAL ERROR OCCURED.")
        raise

    save_results(filename, results)

def save_results(filename, results):
    with open(filename, "wb") as file:
        pickle.dump(results, file)

if __name__ == "__main__":
    jonathan_test_1()