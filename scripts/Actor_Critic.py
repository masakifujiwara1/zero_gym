import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, action_size=2):
        super().__init__()
        # self.l1 = nn.Linear(4, 128)
        self.l1 = nn.Linear(3, 128)
        self.l2 = nn.Linear(128, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.from_numpy(x.astype(np.float32)).clone()
        x = F.relu(self.l1(x))
        # x = self.softmax(self.l2(x))
        x = 2 * torch.tanh(self.l2(x))
        return x

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.from_numpy(x.astype(np.float32)).clone()
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.memory = []
        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()
        self.optimizer_pi = torch.optim.Adam(self.pi.parameters(), self.lr_pi)
        self.optimizer_v = torch.optim.Adam(self.v.parameters(), self.lr_v)
        self.mse = nn.MSELoss()

    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        # print(probs.data)
        probs = probs[0]
        # print(probs.data.sum(), "{:.10g}".format(probs.data[0]), "{:.10g}".format(probs.data[1]))
        # probs = probs.data / probs.data.sum()
        action = np.random.choice(len(probs), p=probs.detach().numpy().copy())
        return action, probs[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self, state, action_prob, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # self.vの損失
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        # target.detach()
        v = self.v(state)
        loss_v = self.mse(v, target)

        # self.piの損失
        delta = target.detach() - v
        # delta.detach()
        loss_pi = -torch.log(action_prob) * delta.detach()
        
        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()

    def model_save(self, path):
        torch.save(self.pi.state_dict(), path + "model_actor-critic.pth")

    def model_load(self, path):
        self.pi.load_state_dict(torch.load(path + "model_actor-critic.pth"))