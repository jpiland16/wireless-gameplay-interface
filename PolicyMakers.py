from Parameters import ParameterSet
from GameElements import Policy, PolicyMaker

class ExamplePolicyMaker(PolicyMaker):
    def __init__(self, params: ParameterSet) -> None:
        super(ExamplePolicyMaker, self).__init__(params, self.get_policy_list)

    def get_policy_list(self):
        return [
            Policy(lambda t: t % self.params.M),
            Policy(lambda t: (t ** 2) % self.params.M)
        ]