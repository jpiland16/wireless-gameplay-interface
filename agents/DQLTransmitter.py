from agents.dql_helpers.dql_agent import DQNAgent
from agents.dql_helpers.event_buffer import Buffer
from agents.dql_helpers.BetterGameSimulator import BetterGameSimulator
from agents.Receivers import ExampleReceiver
from agents.Adversaries import GammaAdversary2
from GameElements import Adversary, GameState, PolicyMaker
from GameParameters import GameParameterSet

from tqdm import tqdm

SHOW_PLOT_AND_DATA = False

if SHOW_PLOT_AND_DATA:
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    import numpy as np
    from sklearn.linear_model import LinearRegression


def train(agent, num_episodes, min_games_per_episode, updates_per_episode, 
        policy_maker, receiver, adversary, internalAdversary, params):

    batch_size = 128
    episode_rewards = []
    episode_switches = []

    print("\n\nTraining the DQN...\n(reward | loss)", end="")

    iter = tqdm(range(num_episodes))

    for episode in iter:
        game_rewards = []
        i = 0

        while (i < min_games_per_episode or
                agent.buffer.current_length < 8 * batch_size):
            gameState = GameState(params, policy_maker.get_policy_list())
            game_reward, switches = BetterGameSimulator(gameState, agent, receiver, adversary, internalAdversary, True).simulate_game()
            game_rewards.append(game_reward)
            i += 1

        for _ in range(updates_per_episode):
            loss = agent.update(batch_size)

        if episode%5 ==4:
            agent.buffer = Buffer(10*batch_size) #should be same as dql_agents
            agent.buffer.current_length = 0

        episode_reward = sum(game_rewards)/len(game_rewards)
        episode_rewards.append(episode_reward)
        iter.set_description(f"{episode_reward:.2f} | {loss:.2f}")

    return episode_rewards

def prepare_for_gameplay(params: GameParameterSet, policy_maker: PolicyMaker, 
        adversary: Adversary, net_params: dict):

    num_episodes = net_params["NUM_EPISODES"]
    updates_per_episode = net_params["UPDATES_PER_EPISODE"]
    min_games_per_episode = net_params["MIN_GAMES_PER_EPISODE"]
    num_simulated_games = net_params["NUM_SIMULATED_GAMES"]

    stateSize = 2 * params.N + 2

    transmitter = DQNAgent(stateSize, params.N)
    receiver = ExampleReceiver()
    internal_adversary = GammaAdversary2()

    rewards = train(transmitter, num_episodes, min_games_per_episode, 
        updates_per_episode, policy_maker, receiver, adversary, 
        internal_adversary, params)

    rewards2 = []
    switches = []

    for _ in range(num_simulated_games):
        game_state = GameState(params, policy_maker.get_policy_list())
        reward, switch = BetterGameSimulator(game_state, transmitter, receiver, 
            adversary, internal_adversary, False).simulate_game()
        rewards2.append(reward)
        switches.append(switch)

    if SHOW_PLOT_AND_DATA:

        X = np.array([i for i in range(num_episodes)]).reshape((-1, 1))
        print(sum(rewards[-10:])/10)
        print(sum(rewards2[-10:])/ 10)
        print(sum(switches)/ 10)

        Y = np.array(rewards)
        #Z = np.array(switches)
        model = LinearRegression()  # create object for the class
        model.fit(X, Y)  # perform linear regression
        Y_pred = model.predict(X)  # make predictions

        r_sq = model.score(X, Y)
        print('coefficient of determination:', r_sq)
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)


        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.show()

    print()

    return DQLTransmitterHelper(transmitter, internal_adversary, policy_maker,
        params)

class DQLTransmitterHelper():
    def __init__(self, transmitter: DQNAgent, internal_adversary: Adversary,
            policy_maker: PolicyMaker, params: GameParameterSet):

        self.transmitter = transmitter

        game_state = GameState(params, policy_maker.get_policy_list())

        self.internal_simulation = BetterGameSimulator(game_state, None, None, 
            None, internal_adversary, False)

        self.nn_input = self.internal_simulation.get_current_state_as_list()

        self.start_policy = 0
        self.last_action = self.start_policy

    def get_policy(self, game_state: GameState):

        previous = game_state.rounds[-1]
        
        _, _, self.nn_input, _ = self.internal_simulation.continue_time(
            self.last_action, previous.transmission_band, 
            previous.adversary_guess, previous.adversary_guess
        )

        next_action = self.transmitter.get_policy(self.nn_input)
        self.last_action = next_action

        # Always communicate (we are not penalized for this though)
        return next_action, True