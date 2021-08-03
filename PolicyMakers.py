import math
import random

from GameParameters import GameParameterSet
from GameElements import Policy, PolicyMaker

class ExamplePolicyMaker(PolicyMaker):
    def __init__(self, params: GameParameterSet) -> None:
        super(ExamplePolicyMaker, self).__init__(params, self.get_policy_list)

    def get_policy_list(self):
        return [
            Policy(lambda t: t % self.params.M, "t % M"),
            Policy(lambda t: (t ** 2) % self.params.M, "t^2 % M")
        ]

class RandomDeterministicPolicyMaker(PolicyMaker):
    def __init__(self, params: GameParameterSet) -> None:
        self.random_sequence_set = [
            [ random.randint(0, params.M - 1) for _ in range(params.T) ] 
                for _ in range(params.N)
        ]
        super(RandomDeterministicPolicyMaker, self).__init__(
            params, self.get_policy_list)

    def get_policy_from_seq_index(self, seq_index: int):
        return Policy(self.random_sequence_set[seq_index].__getitem__,
            f"Random Deterministic Policy #{seq_index}")

    def get_policy_list(self):
        return [self.get_policy_from_seq_index(i) for i in range(self.params.N)]
