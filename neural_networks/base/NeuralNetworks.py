import pickle
from datetime import datetime

# PyTorch imports
import torch
from torch import nn

# Personal code
from Util import confirm

# Neural network imports
from neural_networks.jonathan.SimpleRNNs import SimpleRNN_Adversary

# Check to see if we can use the GPU
device = torch.device('cpu')

def do_gpu_check(default_device):
    global device
    if default_device != "CPU" \
        and torch.cuda.is_available() \
            and (default_device == "GPU" or confirm("\nGPU found! Use GPU?")):
        device = torch.device('cuda')
        print("\nUsing GPU...")
    else:
        print("\nUsing CPU...")

def get_device():
    return device

device = get_device()


class GameAgent():
    """
    A wrapper class for a game player that can perform one
    role within the game. These roles include:
     - Transmitter
     - Receiver
     - Adversary
     - PolicyMaker
     - Value/policy estimator (i.e., the sidekick to MCTS)
    and must be specified in order to avoid errors.
    """
    def __init__(self, role: str, name: str="", nnet: nn.Module=None):
        self.nnet = nnet
        self.role = role
        self.name = name

# ------------------------------------------------------------------------------
# NEURAL NETWORK CLASSES -------------------------------------------------------
# ------------------------------------------------------------------------------

'''
Add neural network agents here that you wish to be trained/used in the game.
Format: (nnet: nn.Module, role: str, name: str)
'''

untrained_networks = [
    GameAgent("Adversary", "Jonathan's Example Adversary", SimpleRNN_Adversary)
]

# ------------------------------------------------------------------------------
# END NEURAL NETWORK CLASSES ---------------------------------------------------
# ------------------------------------------------------------------------------

'''
Code for retrieving pretrained models 
'''

def get_networks_from_disk():
    try:
        nets = pickle.load(open('nnets.pkl', 'rb'))
        return nets
    except:
        return []

def save_networks_to_disk():
    with open('nnets.pkl', 'wb') as file:
        pickle.dump(trained_networks, file)

trained_networks = get_networks_from_disk()

def get_available_networks():
    return untrained_networks + trained_networks

def add_trained_model(agent: GameAgent):
    trained_networks.append(agent)
    agent.name += " - trained {:%Y.%m.%d %H-%M-%S}".format(datetime.now())
