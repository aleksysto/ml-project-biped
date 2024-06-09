import gymnasium as gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from Agent.Agent import Agent
# HYPERPARAMS (unused are commented out):
# weight_decay = 0.0001
# SIGMA = 0.1
TAU = 0.005
CRITIC_LR = 0.0001
ACTOR_LR = 0.0001
LR = 0.0005
GAMMA = 0.99
EPISODES = 2000
STD_DEV = 0.1
BATCH_SIZE = 128
DELAY_INTERVAL = 10
C_VALUE = 0.3  # for clipping noise
# OU_NOISE = 1  # change
# BUFFER = 1  # change
LOAD_MODELS = False


env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
# env = gym.make("Pendulum-v1", render_mode="human")

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print("Actions: ", num_actions, "\n",
      "States: ", num_states, "\n")

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
agent = Agent(
    states=num_states,
    actions=num_actions,
    batch_size=BATCH_SIZE,
    lr=LR,
    gamma=GAMMA,
    tau=TAU,
    delay_interval=DELAY_INTERVAL,
    std_dev=STD_DEV,
    c=C_VALUE,
    memory_len=290000)

RANDOM_STEPS = 3000
STEPS = 0
SOLVED = False

reward_window = deque(maxlen=50)
total_rewards = []
avg_rewards = []
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        STEPS += 1
        if STEPS < RANDOM_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.noise_action(state)
        next, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        agent.save(state, action, reward, next, done)

        agent.replay(STEPS)
        if terminated or truncated:
            done = True
        state = next

    total_rewards.append(total_reward)
    reward_window.append(total_reward)
    avg_reward = np.mean(reward_window)
    avg_rewards.append(avg_reward)

    print("Episode: ", episode,
          " Total reward: ", total_reward,
          " Avg reward: ", avg_reward,
          " Buffer size: ", agent.memory.counter)

    if avg_reward >= 300:
        SOLVED = True
        print("Solved!")

        agent.actor.save_weights("actor_weights.weights.h5")
        agent.actor_target.save_weights(
            "actor_target_weights.weights.h5")

        agent.critic_1.save_weights(
            "critic_1_weights.weights.h5")
        agent.critic_2.save_weights(
            "critic_2_weights.weights.h5")

        agent.critic_1_target.save_weights(
            "critic_1_target_weights.weights.h5")
        agent.critic_2_target.save_weights(
            "critic_2_target_weights.weights.h5")

if not SOLVED:
    agent.actor.save_weights("actor_weights_UNSOLVED.weights.h5")
    agent.actor_target.save_weights(
        "actor_target_weights_UNSOLVED.weights.h5")

    agent.critic_1.save_weights(
        "critic_1_weights_UNSOLVED.weights.h5")
    agent.critic_2.save_weights(
        "critic_2_weights_UNSOLVED.weights.h5")

    agent.critic_1_target.save_weights(
        "critic_1_target_weights_UNSOLVED.weights.h5")
    agent.critic_2_target.save_weights(
        "critic_2_target_weights_UNSOLVED.weights.h5")

plt.plot(avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
