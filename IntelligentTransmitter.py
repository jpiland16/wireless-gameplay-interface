from GameParameters import GameParameterSet
from math import log2
import random
from copy import deepcopy

from GameElements import Policy, Round, Transmitter, GameState
from Adversaries import GammaAdversary

class IntelligentTransmitter(Transmitter):

    def get_expected_rewards(self, game_state: GameState, 
            time_future: int, last_policy: int) -> list:
        
        game_state = deepcopy(game_state)
        game_state.t += time_future
        
        # Calculate the expected immediate reward
        adversary_policy_max_mask = self.internal_adversary.get_policy_max_mask(
            game_state)

        rewards = [ ]

        for policy_choice in range(game_state.params.N):

            # Note, last_policy == -1 signifies the start of the game
            policy_changed = (policy_choice != last_policy 
                and last_policy != -1)

            adversary_total_options = sum(adversary_policy_max_mask)

            # 1 correct option if this policy has the max value, 0 otherwise
            adversary_correct_options = \
                adversary_policy_max_mask[policy_choice]

            probability_adversary_correct = (adversary_correct_options 
                / adversary_total_options)

            expected_reward = ( 
                (1 - probability_adversary_correct) * game_state.params.R1
                # Assume always communicate, so only need to check if policy
                # has changed
                - int(policy_changed) * (game_state.params.R2 + 
                    game_state.params.R3 * log2(game_state.params.N))
            )

            if time_future < self.max_lookahead:

                game_state.rounds.append(
                    Round(game_state.policy_list[policy_choice].get_bandwidth(
                        game_state.t), None, None)
                )

                expected_reward += max(self.get_expected_rewards(game_state, 
                    time_future + 1, policy_choice))
            
            rewards.append(expected_reward)

        return rewards            

    def get_best_policy_index(self, game_state: GameState, 
            time_future: int = 0, last_policy: int = -1) -> int:

        policy_rewards = self.get_expected_rewards(game_state, time_future, 
            last_policy)

        max_value = max(policy_rewards)
        max_value_mask = [ 1 if reward == max_value else 0 
            for reward in policy_rewards ]
        indices = [ i for i in range(len(policy_rewards)) ]

        # When two or more policies have the same value, 
        # don't always choose the one with the lowest index
        policy_id = random.choices(indices, weights=max_value_mask)[0]

        return policy_id

    def policy_selector_function(self, game_state: GameState) -> int:
        
        new_policy = self.get_best_policy_index(game_state, 0, self.last_policy)
        policy_changed = (new_policy != self.last_policy)

        # Note, we are always communicating if the policy changes
        return new_policy, policy_changed

    def __init__(self, game_params: GameParameterSet, 
            policy_list: 'list[Policy]') -> None:

        self.internal_adversary = GammaAdversary()
        self.max_lookahead = 2
        self.last_policy = self.get_best_policy_index(
                GameState(game_params, policy_list, None))

        super().__init__(self.policy_selector_function, 
            start_policy = self.last_policy)
        
        