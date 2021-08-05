
from GameElements import *

class BetterGameSimulator:

    def __init__(self, GameState, transmitter, receiver, adversary, internalAdversary, training):
        self.GameState = GameState
        self.params = GameState.params
        self.policy_list = GameState.policy_list
        self.transmitter = transmitter
        self.receiver = receiver
        self.adversary = adversary
        self.internalAdversary = internalAdversary
        self.training = training

        "Size of state is 2N+1"
        self.state = {
            "1_hot_last_policy": [0 for _ in range(GameState.params.N)],
            "internal_predictions": [0 for _ in range(GameState.params.N)],
            "ad_blocked": 0,
            "t%": 0,
        }
        self.t = 0
        self.last_policy = -1
        self.switches = 0

    def simulate_game(self) -> int:
        # Initialize the return lists
        gameReward = 0

        # Run the game
        self.nnInput = self.get_current_state_as_list()

        done = False
        while not done:
            item = self.advance_time()
            action, reward, next_state, done = item
            if done:
                for i in range(5):
                    self.transmitter.buffer.push(self.nnInput, action, 0, next_state, 0)
            else:
                self.transmitter.buffer.push(self.nnInput, action, reward, next_state, 1)


            state = next_state
            self.nnInput = state
            gameReward += reward

        return gameReward, self.switches

    def get_current_state_as_list(self):

        state_list = [ ]
        
        for value in self.state.values():
            if type(value) is list:
                for element in value:
                    state_list.append(element)
            else:
                state_list.append(value)

        return state_list

    def advance_time(self) -> 'tuple[int, int, list, bool]':
        """
        Advances the time in the game, and calls on each player to make a
        move.
        Returns action, reward, nextState, and whether the game is done.
        """

        t = self.t
        # Select policy --------------------------------
        action = self.transmitter.get_policy(self.nnInput, training=self.training)
        # Transmit based on the selected policy --------------------------------
        policy = self.policy_list[action]

        transmission_band = policy.get_bandwidth(t)
        receiver_guess = transmission_band
        adversary_guess = self.adversary.predict_policy(self.GameState)

        return self.continue_time(action, transmission_band, receiver_guess, 
            adversary_guess)

    def continue_time(self, action, transmission_band, 
            receiver_guess, adversary_guess):
        """
        A continuation of the advance_time method, but can also be used when 
        playing games run by a different simulator.
        """
        reward = 0

        self.GameState.rounds.append(Round(transmission_band, receiver_guess, adversary_guess))

        if (adversary_guess != transmission_band):
            reward += self.params.R1
            self.state["ad_blocked"] = 0
        else:
            self.state["ad_blocked"] = 1
        if action == self.last_policy:
            reward += self.params.R2
        else:
            self.switches += 1

        # Advance the time ----------------------------------------------------
        self.updateState(action)

        nextState = self.get_current_state_as_list()

        if self.t >= self.params.T:
            return action, reward, nextState, True
        else:
            return action, reward, nextState, False



    def updateState(self,action):
        # Update states, actions, rewards based on what happened in the game

        t = self.t

        self.state["1_hot_last_policy"] = [0 for _ in range(self.params.N)]
        self.state["1_hot_last_policy"][action] = 1
        bands= self.internalAdversary.bandwidth_prediction_vals(self.policy_list, self.GameState.rounds, self.params.M, t)

        for i, policy in enumerate(self.policy_list):
            self.state["internal_predictions"][i] = bands[policy.get_bandwidth(t)]
        self.state["t%"] = t / self.params.T
        self.last_policy = action

        t = t + 1
        self.GameState.t =t
        self.t = t