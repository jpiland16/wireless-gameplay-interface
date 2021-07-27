from GameParameters import GameParameterSet
from GameElements import Adversary, GameState

import torch
from torch import nn

class PriyaRLAdversary(Adversary):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:
        if game_state.t > 0:
            # learn
            pass

        return self.net.predict()

    def get_neural_net_input(game_state: GameState) -> 'list[int]':
        pass

    def __init__(self, params: GameParameterSet, num_layers: int, 
            hidden_dim: int, learning_rate: float, lookback: int) -> None:
        self.policy_choice_history = []
        self.lookback = lookback

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

                self.target_output_vectors = []

            def forward(self, input):
                output, _ = self.rnn(input)
                output = output.contiguous().view(-1, self.hidden_dim)   
                output = self.fc(output)
                sig = nn.Sigmoid()
                output = sig(output)
                return output

            def predict(self, input): 
                output = self(input)
                last_layer = output[-1]
                policy_choice = torch.argmax(last_layer).item()
                return policy_choice

            def train(self, input, game_state: GameState):
                """
                Called once at each timestep.
                """
                
                target_output_vector = []
                actual_policy_choice = game_state.policy_choice_history[-1]

                for k in range(game_state.params.N):
                    if k == actual_policy_choice:
                        target_output_vector += [1.0]
                    # TODO check on the timing here with game_state.t (- 1?)
                    elif game_state.policy_list[k].get_bandwidth(game_state.t) \
                            == game_state.rounds[-1].transmission_band:
                        target_output_vector += [0.25]
                    else:
                        target_output_vector += [0.0]

                self.target_output_vectors.append(target_output_vector)

                output = self(input)

                for _ in range(20):    
                    self.zero_grad()
                    criterion = nn.MSELoss()
                    loss = criterion(output, 
                        self.target_output_vectors[-self.lookback - 1:])
                    loss.backward()
                    self.optimizer.step()

        self.net = RL_RNN(
            num_policies = params.N,
            num_layers = num_layers,
            hidden_dim = hidden_dim,
            learning_rate = learning_rate
        )            

        super().__init__(self.bandwidth_predictor_function)