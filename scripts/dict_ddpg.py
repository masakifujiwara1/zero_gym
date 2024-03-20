from ddpg import *
import gym
import matplotlib.pyplot as plt
import numpy as np

def draw_history(history):
    x = np.arange(0, len(history))
    plt.plot(x, history)
    plt.show()

episodes = 300
sync_interval = 20
env = gym.make("Pendulum-v1", render_mode="human", g=9.81, )
# env = gym.make("CartPole-v1", render_mode="human")
agent = Agent(batch_size=64, path="/home/gym_zero/models/")
reward_history = []
max_reward = 0
load_path = "/home/gym_zero/models/"
agent.model_load(load_path)
# reward_history = []
# render_data = np.array([500, 500, 3])

for episode in range(episodes):
    state = env.reset()
    # env.render()
    state = copy.deepcopy(state[0])
    # state = torch.tensor(state, dtype=torch.float)
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        # print(action)
        action = action.detach().numpy()
        # action = int(action)
        # print(env.last_u)
        next_state, reward, done, info, _ = env.step(action.data)

        # agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    # if episode % sync_interval == 0:
    #     agent.sync_qnet()

    reward_history.append(total_reward)

    print("epidode:" + str(episode) + " reward:" + str(total_reward))

draw_history(reward_history)