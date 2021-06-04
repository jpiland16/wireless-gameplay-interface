from GameElements import Transmitter, GameState

class ExampleTransmitter(Transmitter):

    def policy_selector_function(self, game_state: GameState) -> int:
        # Always choose policy 0 and don't communicate it
        return 0, False

    def __init__(self) -> None:
        super(ExampleTransmitter, self).__init__(self.policy_selector_function, 
            start_policy = 0)

    