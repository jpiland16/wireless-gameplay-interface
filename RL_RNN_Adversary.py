from GameParameters import GameParameterSet
from GameElements import Adversary, GameState

import torch
from torch import nn, Tensor

class PriyaRLAdversary(Adversary):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:

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
        predicted_bandwidth = game_state.policy_list[predicted_policy] \
            .get_bandwidth(game_state.t)

        # We have to keep up with this ourselves (not automatically tracked 
        # by GameElements)
        self.policy_choice_history.append(predicted_policy)

        ### LEARN ###

        self.refresh_neural_net_target(game_state)
        self.net.train(self.past_input_vectors + [now_input], 
            self.target_output_vectors)
        
        # Pull in the most recent timestep for the next time we learn
        self.refresh_neural_net_input(game_state)

        return predicted_bandwidth

    def refresh_neural_net_input(self, game_state: GameState) -> None:

        # Create the vector for one timestep
        vector = [0 for _ in range(2 * game_state.params.N)]
        vector[ (game_state.policy_choice_history[-1] * 2) + 1 ] = 1

        for i in range(game_state.params.N):
            # TODO check on timing with game_state.t here
            vector[2 * i] = game_state.policy_list[i].get_bandwidth(
                game_state.t)

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
        actual_policy_choice = game_state.policy_choice_history[-1]
        actual_bandwidth = game_state.policy_list[actual_policy_choice]\
            .get_bandwidth(game_state.t)

        for k in range(game_state.params.N):
            if k == actual_policy_choice:
                target_output_vector += [1.0]
            # TODO check on the timing here with game_state.t (- 1?)
            elif game_state.policy_list[k].get_bandwidth(game_state.t) \
                    == actual_bandwidth:
                target_output_vector += [0.25]
            else:
                target_output_vector += [0.0]

        self.target_output_vectors.append(target_output_vector)

        if len(self.target_output_vectors) > self.lookback:
            # Remove the oldest element
            self.target_output_vectors.pop(0)

    def __init__(self, num_policies: int) -> None:

        NUM_LAYERS = 1
        LEARN_RATE = 0.001
        LOOKBACK = 20
        HIDDEN_DIM = 16

        self.policy_choice_history = []
        self.past_input_vectors = []
        self.target_output_vectors = []
        self.lookback = LOOKBACK

        class RL_RNN(nn.Module):
            def __init__(self, num_policies: int, num_layers: int, 
                    hidden_dim: int, learning_rate: float):
                """
                Initialize this neural network.
                """
                super().__init__()
                self.input_size = num_policies * 2
                self.num_layers = num_layers
                self.hidden_dim = hidden_dim
                self.output_size = num_policies
                self.rnn = nn.RNN(self.input_size, self.hidden_dim, 
                    self.num_layers, batch_first=True)   
                self.fc = nn.Linear(self.hidden_dim, self.output_size)
                self.optimizer = torch.optim.Adam(self.parameters(), 
                    lr=learning_rate)

            def forward(self, input):
                output, _ = self.rnn(input)
                output = output.contiguous().view(-1, self.hidden_dim)   
                output = self.fc(output)
                sig = nn.Sigmoid()
                output = sig(output)
                return output

            def predict(self, input): 
                # Input is expected to be 2D
                output = self(Tensor([input]))
                last_layer = output[-1]
                policy_choice = torch.argmax(last_layer).item()
                return policy_choice

            def train(self, input, target):
                """
                Called once at each timestep.
                """
                for _ in range(20):   
                    output = self(Tensor([input])) 
                    self.zero_grad()
                    criterion = nn.MSELoss()
                    loss = criterion(output, 
                        Tensor(target))
                    loss.backward()
                    self.optimizer.step()

        self.net = RL_RNN(
            num_policies = num_policies,
            num_layers = NUM_LAYERS,
            hidden_dim = HIDDEN_DIM,
            learning_rate = LEARN_RATE
        )            

        super().__init__(self.bandwidth_predictor_function)