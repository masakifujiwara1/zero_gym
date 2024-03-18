import gym
import matplotlib.pyplot as plt
import numpy as np
from replay import *
# from matplotlib.animation import FuncAnimation

env = gym.make("CartPole-v1", render_mode="human")
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):
    state = env.reset()
    state = state[0].copy()
    done = False

    while not done:
        action = 0
        next_state, reward, done, info, _ = env.step(action)
        # print(type(state), type(next_state))
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

state, action, reward, next_state, done = replay_buffer.get_batch()

print(state.shape)
print(action.shape)
print(reward.shape)
print(next_state.shape)
print(done.shape)