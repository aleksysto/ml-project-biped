import gymnasium as gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from Agent.Agent import Agent
# HYPERPARAMS (unused are commented out):
LOAD_MODELS = False
TAU = 0.002
LR = 0.001
GAMMA = 0.99
EPISODES = 1750
STD_DEV = 0.1
BATCH_SIZE = 128
EPSILON = 1
EPSILON_DECAY = 0.0005
EPSILON_MIN = 0.05
DELAY_INTERVAL = 2

env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
# env = gym.make("Pendulum-v1", render_mode="human")

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print("Actions: ", num_actions, "\n",
      "States: ", num_states, "\n")

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
agent = Agent(states=num_states,
              actions=num_actions,
              batch_size=BATCH_SIZE,
              lr=LR,
              gamma=GAMMA,
              tau=TAU,
              delay_interval=DELAY_INTERVAL,
              memory_len=150000)

RANDOM_STEPS = 25000
SOLVED = False
STEPS = 0
reward_window = deque(maxlen=40)
avg_rewards = []
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    timer = 0
    while not done:
        STEPS += 1
        timer += 1
        if np.random.random() <= EPSILON or STEPS < RANDOM_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.action(state)
        if EPSILON > EPSILON_MIN:
            EPSILON -= EPSILON_DECAY

        next, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        agent.save(state, action, reward, next, done)

        agent.replay(STEPS)
        if terminated or truncated:
            done = True
        state = next
        if timer >= 2000:
            done = True

    reward_window.append(total_reward)
    avg_reward = np.mean(reward_window)
    avg_rewards.append(avg_reward)

    print("Episode: ", episode,
          " Total reward: ", total_reward,
          " Avg reward: ", avg_reward,
          " Steps: ", STEPS)

    if avg_reward >= 300:
        SOLVED = True
        print("Solved!")

        agent.critic.save_weights(
            "critic_1_weights.weights.h5")

if not SOLVED:
    agent.critic.save_weights(
        "critic_1_weights.weights.h5")

plt.plot(avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
