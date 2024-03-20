from ddpg import *
import gym
import matplotlib.pyplot as plt
import numpy as np

def draw_history(history):
    x = np.arange(0, len(history))
    plt.plot(x, history)
    plt.show()

episodes = 3000
sync_interval = 10
env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
# env = gym.make("CartPole-v1", render_mode="human")
agent = Agent(batch_size=64, path="/home/gym_zero/models/")
reward_history = []
max_reward = 0

obs_space = env.observation_space.shape
action_space = env.action_space.shape
max_step = env.spec.max_episode_steps

for episode in range(episodes):
    state = env.reset()
    state = copy.deepcopy(state[0])
    # print(type(state), state.shape)
    done = False
    total_reward = 0
    step = 0

    while not done:
        action = agent.get_action(state)
        # action = action.view(-1, 3)
        # print(action)
        # action = env.observation_space.sample()
        # print(action)
        # action = action.detach().numpy()
        # print(type(action.data))
        next_state, reward, done, info, _ = env.step(action.data)
        # next_state = next_state.reshape([3])
        # next_state = next_state.view(-1, 3)
        # print(type(next_state), next_state)


        # agent.update(state, prob, reward, next_state, done)
        # print(state.size, next_state.size)
        # print(type(next_state))
        agent.add_memory(state, action, next_state, reward, done)
        # agent.add(reward, prob)
        state = next_state
        total_reward += reward
        step += 1

        agent.update()

        # if total_reward <= -200:
        #     done = True

        # if step >= 100:
        #     done = True

        # if total_reward >= 500:
        #     done = True
        if max_step < step:
            done = True

    # print(total_reward.item())

    agent.writer.add_scalar("reward", total_reward.item(), episode)
    
    if episode % sync_interval == 0:
        agent.sync_net()

    # agent.update(state, prob, reward, )
    reward_history.append(total_reward)

    # if total_reward >= max_reward:
    #     agent.model_save("/home/gym_zero/models/")
    #     max_reward = total_reward

    print("epidode:" + str(episode) + " reward:" + str(total_reward))

# agent.model_save("/home/gym_zero/models/")

draw_history(reward_history)