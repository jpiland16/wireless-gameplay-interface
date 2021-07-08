import random
from GameElements import Transmitter, GameState
from ShowInfo import show_game_info
from Util import get_integer, confirm

class ExampleTransmitter(Transmitter):

    def policy_selector_function(self, game_state: GameState) -> int:
        # Always choose policy 0 and don't communicate it
        return 0, False

    def __init__(self, num_policies: int) -> None:
        super(ExampleTransmitter, self).__init__(self.policy_selector_function, 
            start_policy = 0)

class HumanTransmitter(Transmitter):

    def policy_selector_function(self, game_state: GameState) -> int:
        # Ask the player to choose policy and communication
        show_game_info(game_state)
        policy = get_integer("New policy [-1 = no change]? (0 - {:d})".format(
            game_state.params.N - 1), min=-1, max=game_state.params.N - 1)
        comm = confirm("Communicate the policy?")
        return policy, comm

    def __init__(self, num_policies: int) -> None:
        super(HumanTransmitter, self).__init__(self.policy_selector_function, 
            start_policy = get_integer("Enter start policy", 
                min=0, max=num_policies - 1))
    
class RandomTransmitter(Transmitter):

    def policy_selector_function(self, game_state: GameState) -> int:
        if random.random() < self.sw_prob:
            new_policy = random.randint(0, game_state.params.N - 2)
            if new_policy >= self.last_policy:
                new_policy += 1
            self.last_policy = new_policy
            return self.last_policy, True
        return self.last_policy, False

    def __init__(self, num_policies: int) -> None:
        self.last_policy = 0
        self.sw_prob = 0.2
        super().__init__(self.policy_selector_function, start_policy = 0)