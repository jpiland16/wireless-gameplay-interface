from GameElements import Transmitter, GameState
from agents.RL_RNN_Adversary import RL_RNN

import torch
from torch import nn, Tensor

class PriyaRLTransmitter(Transmitter):

    def bandwidth_predictor_function(self, game_state: GameState,
            is_initial_run) -> int:

        if not is_initial_run:

            ### LEARN ###

            self.refresh_neural_net_target(game_state)
            self.net.train(self.next_training_input, self.target_output_vectors)
            
            # Pull in the most recent timestep for the next time we learn
            self.refresh_neural_net_input(game_state)

        ### PREDICT ###

        # The 2 here signifies that we do not know if the policy
        # at each index will be chosen or not (rather than a 0 - not chosen
        # or 1 - chosen)
        now_input = [2 for _ in range(2 * game_state.params.N)]
        for i in range(game_state.params.N):
            now_input[2 * i] = game_state.policy_list[i].get_bandwidth(
                game_state.t)
        
        predicted_policy = self.net.predict(self.past_input_vectors 
            + [now_input])

        # Technically this is already stored but hidden in GameState,
        # but it is simpler to use here
        self.policy_choice_history.append(predicted_policy)

        ### SETUP FOR LEARNING ###
        self.next_training_input = self.past_input_vectors + [now_input]

        return predicted_policy

    def refresh_neural_net_input(self, game_state: GameState) -> None:

        # Create the vector for one timestep
        vector = [0 for _ in range(game_state.params.N)]
        
        last_bandwidth = game_state.rounds[-1].transmission_band

        for index, policy in enumerate(game_state.policy_list):
            if policy.get_bandwidth(game_state.t - 1) == last_bandwidth:
                vector[index] = 1

        for i in range(game_state.params.N):
            # TODO check on timing with game_state.t here
            vector[2 * i] = game_state.policy_list[i].get_bandwidth(
                game_state.t - 1)

        if self.past_input_vectors == None:
            self.past_input_vectors == [vector]
        else:
            self.past_input_vectors.append(vector)
            if len(self.past_input_vectors) >= self.lookback:
                # Note that this list must be kept 1 shorter than 
                # target_output_vectors

                # Remove the oldest element
                self.past_input_vectors.pop(0)

    def refresh_neural_net_target(self, game_state: GameState):

        target_output_vector = []

        actual_bandwidth = game_state.rounds[-1].transmission_band

        for k in range(game_state.params.N):
            # TODO check on the timing here with game_state.t (- 1?)
            if game_state.policy_list[k].get_bandwidth(game_state.t - 1) \
                    == actual_bandwidth:
                target_output_vector += [1.0]
            else:
                target_output_vector += [0.0]

        self.target_output_vectors.append(target_output_vector)

        if len(self.target_output_vectors) > self.lookback:
            # Remove the oldest element
            self.target_output_vectors.pop(0)

    def __init__(self, num_policies: int, net_params: dict) -> None:
        """
        Initialize this adversary and its neural network.
        Note: net_params should be a dict containing the following keys:
         - NUM_LAYERS
         - LEARNING_RATE
         - LOOKBACK
         - HIDDEN_DIM
         - REPETITIONS
        """
        self.policy_choice_history = []
        self.past_input_vectors = []
        self.target_output_vectors = []
        self.lookback = net_params["LOOKBACK"]

        self.net = RL_RNN(
            num_policies = num_policies,
            num_layers = net_params["NUM_LAYERS"],
            hidden_dim = net_params["HIDDEN_DIM"],
            learning_rate = net_params["LEARNING_RATE"],
            repetitions = net_params["REPETITIONS"]
        )            

        super().__init__()