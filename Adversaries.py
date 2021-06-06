from GameElements import Adversary, GameState
from ShowInfo import show_game_info
import random

class ExampleAdversary(Adversary):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:
        # Guess a random band (0 to M-1)
        return random.randint(0, game_state.params.M - 1)

    def __init__(self) -> None:
        super(ExampleAdversary, self).__init__(
            self.bandwidth_predictor_function)


class HumanAdversary(Adversary):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:
        # Ask the player to predict the bandwidth based on game info
        show_game_info(game_state)
        return int(input("Predicted band? (0 - {:d}) > ".format(
            game_state.params.M - 1)))

    def __init__(self) -> None:
        super(HumanAdversary, self).__init__(
            self.bandwidth_predictor_function)