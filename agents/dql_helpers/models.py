import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        sizeFirstHidden = 8
        sizeSecondHidden = 6

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, sizeFirstHidden),
            nn.ReLU(),
            nn.Linear(sizeFirstHidden, sizeSecondHidden),
            nn.ReLU(),
            nn.Linear(sizeSecondHidden, self.output_dim)
        )


    def  forward(self, state):
        "Input's parameters for state, outputs a vector of n qvals"
        qvals = self.fc(state)
        return qvals