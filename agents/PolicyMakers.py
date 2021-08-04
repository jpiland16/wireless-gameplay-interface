import math
import random
import warnings

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

class SimilarPolicyMaker(PolicyMaker):

    def __init__(self, params: GameParameterSet, similarity_score: float):
        """
        Generates 'similar' patterns according to the following definition:

        For each bandwidth B at a given timestep, the number of policies that 
        have that bandwidth is defined as Count(B). The similarity score is 
        defined as the sum of (Count(B) - 1) over all B, divided by N,
        averaged over all timesteps.

        If there are N <= B policies, the minimum similarity score is 0 and 
        the maximum score is (N - 1) / N.

        If N > B, the minimum is increased to (N - B) / N.

        The similarity at each timestep varies according to a normal 
        distribution with sigma = 0.1.
        """
        self.params = params
        self.similarity_score = similarity_score

        self.min_similarity = max(0, params.N - params.M) / params.N
        self.max_similarity = (params.N - 1) / params.N

        if similarity_score < self.min_similarity:
            warnings.warn(f"Similarity score {similarity_score} is less " +
                          f"than the minimum {self.min_similarity}. Using " +
                           "minumum instead.")
            self.similarity_score = self.min_similarity

        if similarity_score > self.max_similarity:
            warnings.warn(f"Similarity score {similarity_score} is greater " +
                          f"than the maximum {self.max_similarity}. Using " +
                           "maximum instead.")
            self.similarity_score = self.max_similarity

        self.generate_policies()

        super().__init__(params, self.get_policy_list)

    def get_policy_list(self):
        return [Policy(self.sequences[i].__getitem__, f"Similar Policy #{i}") 
            for i in range(self.params.N)]

    def generate_policies(self):
        """
        Algorithm:

        Suppose we have 10 policies, 10 bands, and want a similarity of 0.5.
        This means that we need 6 policies to have the same band (or 4 and 3,
        or 2 and 5, or 2, 3, and 3 ...)

        """

        self.sequences = [[] for _ in range(self.params.N)]

        for _ in range(self.params.T):
            
            # UNCOMMENT THIS IF YOU WANT VARYING SIMILARITY
            # this_time_similarity = max(self.min_similarity, 
            #     min(random.normalvariate(self.similarity_score, 0.1), 
            #     self.max_similarity))

            this_time_similarity = self.similarity_score

            num_matching_bands = round(this_time_similarity * self.params.N)
            
            policy_not_taken = [1] * self.params.N
            bandwidth_order = [i for i in range(self.params.M)]
            random.shuffle(bandwidth_order)
            bandwidth_selection_index = 0

            # Divide the num_matching_bands into one or more partitions
            max_num_partitions = min(self.params.N - num_matching_bands, 
                num_matching_bands)
            
            if max_num_partitions > 0:

                partition_count = random.randint(1, max_num_partitions)

                matching_policy_index_sets = [[] for _ in range(partition_count)]

                for index in range(partition_count):
                    # Initialize the matching lists with at least two policies
                    for _ in range(2):
                        policy_index = random.choices(
                            [i for i in range(self.params.N)], 
                            weights=policy_not_taken
                        )[0]

                        policy_not_taken[policy_index] = 0
                        matching_policy_index_sets[index].append(policy_index)
                            
                while (sum([len(lst) for lst in matching_policy_index_sets])
                    - partition_count) < num_matching_bands:

                    policy_index = random.choices(
                            [i for i in range(self.params.N)], 
                            weights=policy_not_taken
                        )[0]

                    policy_not_taken[policy_index] = 0

                    group_index = random.randint(0, partition_count - 1)

                    matching_policy_index_sets[group_index].append(policy_index)

                for group in matching_policy_index_sets:
                    
                    for policy_index in group:

                        self.sequences[policy_index].append(bandwidth_order
                            [bandwidth_selection_index])

                    bandwidth_selection_index += 1

            for index, not_taken in enumerate(policy_not_taken):

                if bool(not_taken):

                    self.sequences[index].append(bandwidth_order
                        [bandwidth_selection_index])

                    bandwidth_selection_index += 1
    
    def calculate_similarity_score(self):

        total_similarity = 0

        for t in range(self.params.T):

            band_count = [0] * self.params.M

            for sequence in self.sequences:
                band_count[sequence[t]] += 1

            total_similarity += sum([max(0, count - 1) 
                for count in band_count]) / self.params.N
        
        return total_similarity / self.params.T

