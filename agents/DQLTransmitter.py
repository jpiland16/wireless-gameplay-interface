from agents.dql_helpers.dql_agent import DQNAgent
from agents.dql_helpers.event_buffer import Buffer
from agents.dql_helpers.BetterGameSimulator import BetterGameSimulator
from agents.Receivers import ExampleReceiver
from agents.Adversaries import GammaAdversary2
from GameElements import Adversary, GameState, PolicyMaker
from GameParameters import GameParameterSet

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from sklearn.linear_model import LinearRegression


def train(agent, num_episodes, min_games_per_episode, updates_per_episode, 
        policy_maker, receiver, adversary, internalAdversary, params):

    batch_size = 128
    episode_rewards = []
    episode_switches = []

    for episode in range(num_episodes):
        game_rewards = []
        i = 0

        while (i < min_games_per_episode or
                agent.buffer.current_length < 8 * batch_size):
            gameState = GameState(params, policy_maker.get_policy_list())
            game_reward, switches = BetterGameSimulator(gameState, agent, receiver, adversary, internalAdversary, True).simulate_game()
            game_rewards.append(game_reward)
            i += 1

        for k in range(updates_per_episode):
            loss = agent.update(batch_size)


        if episode%5 ==4:
            agent.buffer = Buffer(10*batch_size) #should be same as dql_agents
            agent.buffer.current_length = 0

        episode_reward = sum(game_rewards)/len(game_rewards)
        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))
    return episode_rewards

def prepare_for_gameplay(params: GameParameterSet, policy_maker: PolicyMaker, 
        adversary: Adversary):

    NUM_EPISODES = 3
    UPDATES_PER_EPISODE = 50
    MIN_GAMES_PER_EPISODE = 3
    NUM_SIMULATED_GAMES = 10

    stateSize = 2 * params.N + 2

    transmitter = DQNAgent(stateSize, params.N)
    receiver = ExampleReceiver()
    internal_adversary = GammaAdversary2()

    rewards = train(transmitter, NUM_EPISODES, MIN_GAMES_PER_EPISODE, 
        UPDATES_PER_EPISODE, policy_maker, receiver, adversary, 
        internal_adversary, params)

    rewards2 = []
    switches = []

    for _ in range(NUM_SIMULATED_GAMES):
        game_state = GameState(params, policy_maker.get_policy_list())
        reward, switch = BetterGameSimulator(game_state, transmitter, receiver, 
            adversary, internal_adversary, False).simulate_game()
        rewards2.append(reward)
        switches.append(switch)


    X = np.array([i for i in range(NUM_EPISODES)]).reshape((-1, 1))
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

    return transmitter
