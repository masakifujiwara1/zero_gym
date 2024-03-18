from DQN_gpu import *
import gym
import matplotlib.pyplot as plt
import numpy as np

def draw_history(history):
    x = np.arange(0, len(history))
    plt.plot(x, history)
    plt.show()

episodes = 300
sync_interval = 20
env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("CartPole-v1", render_mode="human")
agent = DQNAgent()
reward_history = []

save_path = "/home/gym_zero/models/"

for episode in range(episodes):
    state = env.reset()
    state = copy.deepcopy(state[0])
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        # print(action)
        action = int(action)
        next_state, reward, done, info, _ = env.step(action)

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)

    print("epidode:" + str(episode) + " reward:" + str(total_reward))

draw_history(reward_history)

agent.model_save(save_path)