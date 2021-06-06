from GameElements import Transmitter, GameState
from ShowInfo import show_game_info

class ExampleTransmitter(Transmitter):

    def policy_selector_function(self, game_state: GameState) -> int:
        # Always choose policy 0 and don't communicate it
        return 0, False

    def __init__(self) -> None:
        super(ExampleTransmitter, self).__init__(self.policy_selector_function, 
            start_policy = 0)

class HumanTransmitter(Transmitter):

    def policy_selector_function(self, game_state: GameState) -> int:
        # Ask the player to choose policy and communication
        show_game_info(game_state)
        policy = int(input("New policy [-1 = no change]? (0 - {:d}) > ".format(
            game_state.params.N - 1)))
        comm = input("Communicate the policy? (Y/n) > ")
        return policy, (comm == 'Y' or comm == 'y' or comm == 'yes')

    def __init__(self) -> None:
        super(HumanTransmitter, self).__init__(self.policy_selector_function, 
            start_policy = int(input("Enter start policy: ")))
    