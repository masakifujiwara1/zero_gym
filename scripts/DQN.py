import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay import *
import numpy as np

class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.from_numpy(x.astype(np.float32)).clone()
        x1 = F.relu(self.l1(x))
        x2 = F.relu(self.l2(x1))
        x3 = self.l3(x2)
        return x3

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        # self.optimizer.setup(self.qnet) # qnetを設定

        self.loss = nn.MSELoss()

        print(next(self.qnet.parameters()).is_cuda)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :] # バッチの次元を追加
            qs = self.qnet(state)
            return qs.data.argmax()

    def dict(self, state):
        qs = self.qnet(state)
        # action, _ = qs.max(axis=1)
        return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return 

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q, _ = next_qs.max(axis=1)
        next_q = next_q.detach().numpy()
        target = reward + (1 - done) * self.gamma * next_q
        target = torch.from_numpy(target.astype(np.float32)).clone()

        loss = self.loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()   

    def model_save(self, path):
        torch.save(self.qnet.state_dict(), path + "model.pth")

    def model_load(self, path):
        self.qnet.load_state_dict(torch.load(path + "model_gpu.pth"))