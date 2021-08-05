import numpy as np
from GameElements import Round, Adversary, GameState, Policy
from ShowInfo import show_game_info
from Util import get_integer
import random

from agents.RL_RNN_Adversary import PriyaRL_NoPolicy, PriyaRL_WithPolicy

class ExampleAdversary(Adversary):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:
        # Guess a random band (0 to M-1)
        # return random.randint(0, game_state.params.M - 1)

        # Guess a random policy (0 to N - 1)
        return random.choice(game_state.policy_list).get_bandwidth(game_state.t)

    def __init__(self) -> None:
        super(ExampleAdversary, self).__init__(
            self.bandwidth_predictor_function)


class HumanAdversary(Adversary):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:
        # Ask the player to predict the bandwidth based on game info
        show_game_info(game_state)
        return get_integer("Predicted band? (0 - {:d})".format(
            game_state.params.M - 1), min=0, max=game_state.params.M - 1)

    def __init__(self) -> None:
        super(HumanAdversary, self).__init__(
            self.bandwidth_predictor_function)

class GammaAdversary(Adversary):

    def get_policy_value(self, policy: Policy, rounds: 'list[Round]'):

        lo_lookback_index = max(0, len(rounds) - self.max_lookback)
        lookback_range = range(lo_lookback_index, len(rounds))

        state_values = [self.gamma ** (len(rounds) - t) 
            for t in lookback_range]

        policy_predictions = [policy.get_bandwidth(t) 
            for t in lookback_range]
        
        value = sum([state_values[t] 
            if policy_predictions[t] == \
                rounds[t + lo_lookback_index].transmission_band else 0 
            for t in range(len(lookback_range))])

        return value

    def get_policy_max_mask(self, game_state: GameState) -> 'list[int]':
        """
        Return a 1 for policies that have the maximum value, and a 0 otherwise.
        """

        policy_values = [self.get_policy_value(policy, game_state.rounds) 
            for policy in game_state.policy_list] 

        max_value = max(policy_values)
        max_value_mask = [ 1 if v == max_value else 0 for v in policy_values ]

        return max_value_mask

    def bandwidth_prediction_function(self, game_state: GameState) -> int:
        
        indices = [ i for i in range(game_state.params.N) ]

        # When two or more policies have the same value, 
        # don't always choose the one with the lowest index
        policy_id = random.choices(indices, weights=self.get_policy_max_mask(
            game_state))[0]

        return game_state.policy_list[policy_id].get_bandwidth(game_state.t)

    def __init__(self) -> None:
        self.gamma = 0.3
        self.max_lookback = 5
        super().__init__(self.bandwidth_prediction_function)

class GammaAdversary2(Adversary):

    def get_policy_value(self, policy: Policy, rounds: 'list[Round]'):

        lo_lookback_index = max(0, len(rounds) - self.max_lookback)
        lookback_range = range(lo_lookback_index, len(rounds))

        state_values = [self.gamma ** (len(rounds) - t) 
            for t in lookback_range]

        policy_predictions = [policy.get_bandwidth(t) 
            for t in lookback_range]
        
        value = sum([state_values[t] 
            if policy_predictions[t] == \
                rounds[t + lo_lookback_index].transmission_band else 0 
            for t in range(len(lookback_range))])

        return value

    def bandwidth_prediction_function(self, game_state: GameState) -> int:
        bands = [0 for _ in range(game_state.params.M)]
        for policy in game_state.policy_list:
            bands[policy.get_bandwidth(game_state.t)] += \
                self.get_policy_value(policy, game_state.rounds)

        max_band_value = max(bands)

        band_indices = [ i for i in range(len(bands))]
        band_max_mask = [1 if band_value == max_band_value else 0 
            for band_value in bands]

        band_choice = random.choices(band_indices, weights=band_max_mask)[0]

        return band_choice
        
    def bandwidth_prediction_vals(self, policy_list, rounds, M, t) -> int:
        bands = [0 for _ in range(M)]
        for policy in policy_list:
            bands[policy.get_bandwidth(t)] += self.get_policy_value(policy, rounds)

        return bands

    def __init__(self) -> None:
        self.gamma = 0.3
        self.max_lookback = 3
        super().__init__(self.bandwidth_prediction_function)