from GameElements import Receiver, GameState

class ExampleReceiver(Receiver):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:
        # Guess based on whatever the last policy was
        predicted_policy = game_state.policy_list \
            [self.last_policy_communicated]
        return predicted_policy.get_bandwidth(game_state.t)

    def communication_channel(self, policy) -> None:
        self.last_policy_communicated = policy

    def __init__(self) -> None:
        super(ExampleReceiver, self).__init__(self.bandwidth_predictor_function, 
            self.communication_channel)