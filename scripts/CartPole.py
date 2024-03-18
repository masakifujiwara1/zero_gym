import gym
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.animation import FuncAnimation

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()
# draw_array = env.render()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, info, _ = env.step(action)
env.close()
