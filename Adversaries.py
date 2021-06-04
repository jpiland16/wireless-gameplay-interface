from GameElements import Adversary, GameState
import random

class ExampleAdversary(Adversary):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:
        # Guess a random band (0 to M-1)
        return random.randint(0, game_state.params.M - 1)

    def __init__(self) -> None:
        super(ExampleAdversary, self).__init__(
            self.bandwidth_predictor_function)