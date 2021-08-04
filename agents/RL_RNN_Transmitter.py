from GameParameters import GameParameterSet
from GameElements import Policy, Transmitter, GameState
from agents.RL_RNN_Adversary import RL_RNN

class PriyaRLTransmitter(Transmitter):

    def policy_predictor_function(self, game_state: GameState,
            is_initial_run: bool = False) -> int:

        if not is_initial_run:

            ### LEARN ###

            self.refresh_neural_net_target(game_state)
            self.net.train(self.next_training_input, self.target_output_vectors)
            
            # Pull in the most recent timestep for the next time we learn
            self.refresh_neural_net_input(game_state)

        ### PREDICT ###

        # The 2 here signifies that we do not know if the policy
        # at each index will be chosen or not (rather than a 0 - not chosen
        # or 1 - chosen)
        now_input = [2 for _ in range(2 * game_state.params.N)]
        for i in range(game_state.params.N):
            now_input[2 * i] = game_state.policy_list[i].get_bandwidth(
                game_state.t)
        
        predicted_policy = self.net.predict(self.past_input_vectors 
            + [now_input])

        # Technically this is already stored but hidden in GameState,
        # but it is simpler to use here
        self.policy_choice_history.append(predicted_policy)

        ### SETUP FOR LEARNING ###
        self.next_training_input = self.past_input_vectors + [now_input]

        return predicted_policy

    def refresh_neural_net_input(self, game_state: GameState) -> None:

        # Create the vector for one timestep
        vector = [0 for _ in range(2 * game_state.params.N)]

        vector[ (self.policy_choice_history[-1] * 2) + 1 ] = 1

        for i in range(game_state.params.N):
            # TODO check on timing with game_state.t here
            vector[2 * i] = game_state.policy_list[i].get_bandwidth(
                game_state.t - 1)

        if self.past_input_vectors == None:
            self.past_input_vectors == [vector]
        else:
            self.past_input_vectors.append(vector)
            if len(self.past_input_vectors) >= self.lookback:
                # Note that this list must be kept 1 shorter than 
                # target_output_vectors

                # Remove the oldest element
                self.past_input_vectors.pop(0)

    def refresh_neural_net_target(self, game_state: GameState):

        target_output_vector = []

        for index, policy in enumerate(game_state.policy_list):

            message_would_be_jammed = (policy.get_bandwidth(game_state.t) == 
                game_state.rounds[-1].adversary_guess)
            
            policy_changed = (len(self.policy_choice_history) > 1 and
                self.policy_choice_history[-2] != index)

            if message_would_be_jammed:
                if policy_changed:
                    target_output_vector += [0.0]
                else:
                    target_output_vector += [0.3]
            else:
                if policy_changed:
                    target_output_vector += [0.7]
                else:
                    target_output_vector += [1.0] 

        self.target_output_vectors.append(target_output_vector)

        if len(self.target_output_vectors) > self.lookback:
            # Remove the oldest element
            self.target_output_vectors.pop(0)

    def policy_and_communication(self, game_state):
        # Communication is necessary, but ignored for scoring
        return self.policy_predictor_function(game_state), True

    def __init__(self, policy_list: 'list[Policy]', net_params: dict) -> None:
        """
        Initialize this adversary and its neural network.
        Note: net_params should be a dict containing the following keys:
         - NUM_LAYERS
         - LEARNING_RATE
         - LOOKBACK
         - HIDDEN_DIM
         - REPETITIONS
        """
        num_policies = len(policy_list)

        self.policy_choice_history = []
        self.past_input_vectors = []
        self.target_output_vectors = []
        self.lookback = net_params["LOOKBACK"]

        self.net = RL_RNN(
            input_size = num_policies * 2,
            output_size = num_policies,
            num_layers = net_params["NUM_LAYERS"],
            hidden_dim = net_params["HIDDEN_DIM"],
            learning_rate = net_params["LEARNING_RATE"],
            repetitions = net_params["REPETITIONS"]
        )            

        temp_params = GameParameterSet(-1, num_policies, -1, -1, -1, -1)
        temp_game_state = GameState(temp_params, policy_list)

        super().__init__(self.policy_and_communication, 
            self.policy_predictor_function(temp_game_state, 
                is_initial_run = True))