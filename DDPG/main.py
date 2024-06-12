import gymnasium as gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from Agent.Agent import Agent
# HYPERPARAMS NOTES:
# changed lr from 0.002 and 0.001 to 0.0001 for both, more stable and overall a tiny bit better, leaving it as is
# changed tau from 0.005 to 0.002 and the results seem better
# changed tau from 0.002 to 0.001 and results went above -100
# changed critic shape and nothing happened yet, will mix with tau 0.005, if that does nothing, change lr too
# changing lr to 0.001 didnt do much
# testing batch size 128 lr 0.0001
# doing previous with 256 batch, tau 0.005, lr 0.001 and OUActionNoise

# 128 batch, tau 0.001, ounoise, critic lr 0.0003, actor lr 0.0001, critic weight decay 0.0001 - new-set-graph
# 128 batch, tau 0.001, ounoise, critic lr 0.0003, actor lr 0.0001, critic weight decay 0.0001 (400-300 neuron)
# 128 batch, tau 0.001, ounoise, critic lr 0.0001, actor lr 0.00005, critic weight decay 0.0001 (400-300 neuron) 
# 128 batch, tau 0.005, ounoise, critic lr 0.0001, actor lr 0.00005, critic weight decay 0.0001 (400-300 neuron) 

# test 0.01 tau, critic learning rate to 0.00005, implement ornstein, add qvalue and gradient monitoring
# add l2 regularization, batch normalization, test 0.98 and 0.95 discount,  
# Sharp decline due to policy deterministic, weak exploration, instability in qval predictions
# HYPERPARAMS (unused are commented out):
# weight_decay = 0.0001
# SIGMA = 0.1
LOAD_MODELS = False
TAU = 0.001
CRITIC_LR = 0.001
ACTOR_LR = 0.0005
GAMMA = 0.99
EPISODES = 1000
STD_DEV = 0.1
BATCH_SIZE = 128
C_VALUE = 0.4  # for clipping noise
BUFFER_SIZE = 100000
# OU_NOISE = 1  # change
# BUFFER = 1  # change


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
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    gamma=GAMMA,
    tau=TAU,
    std_dev=STD_DEV,
    c=C_VALUE,
    memory_len=BUFFER_SIZE)

RANDOM_STEPS = 500
STEPS = 0
SOLVED = False

reward_window = deque(maxlen=40)
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

        agent.replay()
        if terminated or truncated:
            done = True
        state = next

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

        agent.critic.save_weights(
            "critic_weights.weights.h5")
        agent.critic_target.save_weights(
            "critic_target_weights.weights.h5")
        break

if not SOLVED:
    agent.actor.save_weights("actor_weights_UNSOLVED.weights.h5")
    agent.actor_target.save_weights(
        "actor_target_weights_UNSOLVED.weights.h5")

    agent.critic.save_weights(
        "critic_weights_UNSOLVED.weights.h5")
    agent.critic_target.save_weights(
        "critic_target_weights_UNSOLVED.weights.h5")

plt.plot(avg_rewards)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
