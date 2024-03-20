import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torchrl.data import ReplayBuffer, ListStorage
from replay_buffer import *
from torch.utils.tensorboard import SummaryWriter

class ActorNet(nn.Module):
    def __init__(self, action_size=2, obs_space=3):
        super().__init__()
        # self.l1 = nn.Linear(4, 128)
        self.l1 = nn.Linear(obs_space, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_size)
        # self.softmax = nn.Softmax(dim=1)

        self.mean = nn.Linear(300, action_size)
        self.std = nn.Linear(300, action_size)

    def forward(self, x):
        # print(type(x))
        # x = torch.from_numpy(x.astype(np.float32)).clone()
        # print(x.shape)
        x = x.view(-1, 3)
        x = F.relu(self.l1(x))
        # x = self.softmax(self.l2(x))
        x = F.relu(self.l2(x))
        x = 2 * torch.tanh(self.l3(x))
        # mean = self.mean(x)
        # std = torch.exp(self.std(x))
        # action = torch.normal(mean=mean, std=std)
        # x = 2 * torch.tanh(action)
        # print(x.shape)
        return x

class CriticNet(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.l1 = nn.Linear(obs_space, 400)
        self.l2 = nn.Linear(400+1, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, action):
        # x = torch.from_numpy(x.astype(np.float32)).clone()
        # print(x.shape)
        action = action.view(-1, 1)
        # print(action.shape)
        x = F.relu(self.l1(x))
        # print(x.shape, action.shape)
        x = F.relu(self.l2(torch.cat([x, action], dim=1)))
        x = self.l3(x)
        return x

class Agent:
    def __init__(self, batch_size, device="cpu", path="/home/gym_zero/models/"):
        self.gamma = 0.99
        self.lr_pi = 1e-4
        self.lr_v = 1e-3
        self.action_size = 1

        self.memory = []
        self.actor = ActorNet(action_size=self.action_size)
        self.critic = CriticNet(3)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), self.lr_pi)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr_v)
        self.mse = nn.MSELoss()
        self.batch_size = batch_size
        self.device = device

        self.replay_buffer = ReplayBuffer()

        self.path = path
        self.writer = SummaryWriter(log_dir="/home/gym_zero/runs")

        # torch.manual_seed(0)
        # self.rb = ReplayBuffer(storage=ListStorage(max_size=1000), batch_size=self.batch_size)

    def get_action(self, state):
        # if len(self.replay_buffer) < self.batch_size:
        #     return
        state = state[np.newaxis, :]
        state_tensor = torch.tensor(state, dtype=torch.float, device="cpu")
        # print(state.shape)
        action = self.actor(state_tensor)
        return action

    def add_memory(self, *args):
        # data = (state, action, reward, next_state, done)
        # self.rb.add(data)
        self.replay_buffer.append(*args)

    def reset_memory(self):
        self.replay_buffer.reset()

    def sync_net(self):
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.model_save(self.path)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # state = state[np.newaxis, :]
        # next_state = next_state[np.newaxis, :]

        # buffer = self.rb.sample(self.batch_size)
        # print(buffer)

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # print(batch.next_obs)
        obs_batch = torch.tensor(batch.obs, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.float)
        next_obs_batch = torch.tensor(batch.next_obs, device=self.device, dtype=torch.float)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float).unsqueeze(1)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float).unsqueeze(1)

        # print(obs_batch.shape, next_obs_batch.shape)
        # print(next_obs_batch[0])
        # print(next_state.shape)
        # print(type(reward))
        # reward = reward[0]

        # self.vの損失
        # target = reward + self.gamma * self.v(next_state) * (1 - done)
        # target.detach()
        # print(state.shape)
        # v = self.v(state)

        qvalue = self.critic(obs_batch, action_batch)
        next_qvalue = self.critic_target(next_obs_batch, self.actor_target(next_obs_batch))
        target_qvalue = reward_batch + (1 - done_batch) * self.gamma * next_qvalue

        # print(target_qvalue)

        # criticの損失
        loss_critic = self.mse(qvalue, target_qvalue)

        # actorの損失
        loss_actor = -self.critic(obs_batch, self.actor(obs_batch)).mean()
        # delta = target.detach() - v
        # delta.detach()
        # loss_pi = -torch.log(action_prob) * delta.detach()
        
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        loss_actor.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def model_save(self, path):
        torch.save(self.actor.state_dict(), path + "model_actor_ddpg.pth")
        torch.save(self.critic.state_dict(), path + "model_critic_ddpg.pth")

    def model_load(self, path):
        self.actor.load_state_dict(torch.load(path + "model_actor_ddpg.pth"))
        self.critic.load_state_dict(torch.load(path + "model_critic_ddpg.pth"))