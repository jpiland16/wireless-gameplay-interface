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
    