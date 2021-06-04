from Transmitters import *
from Receivers import *
from Adversaries import *
from PolicyMakers import *
from GameElements import *

def simulate_game(params: ParameterSet, policy_maker: PolicyMaker,
    transmitter: Transmitter, receiver: Receiver, adversary: Adversary) -> Game:

        policy_list = policy_maker.get_policy_list()
        game = Game(transmitter, receiver, adversary, policy_list, params)

        # Communicate the initial policy (very important)
        receiver.communicate(transmitter.start_policy)

        while(game.advance_time()):
            pass

        return game

def repeat_game(count: int, params: ParameterSet, policy_maker: PolicyMaker,
    transmitter: Transmitter, receiver: Receiver, adversary: Adversary) -> None:

    for i in range(0, count):
        result = simulate_game(params, policy_maker, transmitter, receiver, 
            adversary)
        score_a = result.state.score_a
        score_b = result.state.score_b
        winner = 'A' if score_a > score_b else \
            'B' if score_b > score_a else '-'
        print("Trial {:3d}  |  Score A: {:4d}  |  Score B: {:4d}  |  Winner: {:1s}"
            .format(i + 1, score_a, score_b, winner ))


def demo():
    params = ParameterSet(
        M = 3,
        N = 2,
        T = 100,
        R1 = 10,
        R2 = 5,
        R3 = 20
    )

    policy_maker = ExamplePolicyMaker(params)
    
    transmitter = ExampleTransmitter()
    receiver = ExampleReceiver()
    adversary = ExampleAdversary()

    repeat_game(10, params, policy_maker, transmitter, receiver, adversary)


def main():
    demo()

if __name__ == '__main__':
    main()