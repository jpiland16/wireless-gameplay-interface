import torch
import torch.nn as nn
import numpy as np

from random import randrange
from agents.dql_helpers.event_buffer import Buffer
from agents.dql_helpers.models import DQN

class DQNAgent:

    def __init__(self, NumObs, NumActions, learning_rate=.001, gamma=.2, buffer_size=100000):

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer = Buffer(buffer_size)

        self.num_policies = NumActions

        self.device = torch.device("cpu")
        self.model = DQN(NumObs, NumActions).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),learning_rate)
        self.MSE_loss = nn.MSELoss()

        #TODO fix starter policy
        self.start_policy = 0




    def get_policy(self, state, training=False, eps=.3):
        state = torch.FloatTensor(state).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        #print(qvals)

        if training and abs(np.random.randn()) < eps:
            return randrange(self.num_policies)
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, notDones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        notDones = torch.tensor(notDones).to(self.device)


        curr_Q= self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]

        #print(max_next_Q)
        expected_Q = rewards + (self.gamma*max_next_Q* notDones)

        loss = self.MSE_loss(curr_Q, expected_Q)
        #print(loss)
        #print(loss.size())
        return loss

    def update(self, batch_size):
        batch = self.buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
