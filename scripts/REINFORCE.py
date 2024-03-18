import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.from_numpy(x.astype(np.float32)).clone()
        x = F.relu(self.l1(x))
        x = self.softmax(self.l2(x))
        return x

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = torch.optim.Adam(self.pi.parameters(), self.lr)

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

    def update(self):
        self.optimizer.zero_grad()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G

        # for reward, prob in self.memory:
            loss += -torch.log(prob) * G
        
        loss.backward()
        self.optimizer.step()
        self.memory = []

    def model_save(self, path):
        torch.save(self.pi.state_dict(), path + "model.pth")

    def model_load(self, path):
        self.pi.load_state_dict(torch.load(path + "model.pth"))