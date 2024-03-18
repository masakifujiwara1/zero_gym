from Actor_Critic import *
import gym
import matplotlib.pyplot as plt
import numpy as np

def draw_history(history):
    x = np.arange(0, len(history))
    plt.plot(x, history)
    plt.show()

episodes = 3000
# sync_interval = 20
env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
# env = gym.make("CartPole-v1", render_mode="human")
agent = Agent()
reward_history = []
max_reward = 0

for episode in range(episodes):
    state = env.reset()
    state = copy.deepcopy(state[0])
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        # print(action)
        action = int(action)
        next_state, reward, done, info, _ = env.step(action)

        agent.update(state, prob, reward, next_state, done)
        # agent.add(reward, prob)
        state = next_state
        total_reward += reward

        if total_reward >= 500:
            done = True
    
    # if episode % sync_interval == 0:
    #     agent.sync_qnet()

    # agent.update(state, prob, reward, )
    reward_history.append(total_reward)

    if total_reward >= max_reward:
        agent.model_save("/home/gym_zero/models/")
        max_reward = total_reward

    print("epidode:" + str(episode) + " reward:" + str(total_reward))

# agent.model_save("/home/gym_zero/models/")

draw_history(reward_history)