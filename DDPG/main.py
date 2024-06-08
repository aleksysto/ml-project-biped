import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from Buffer.Buffer import Buffer
from Networks.Networks import get_actor, get_critic
from OUActionNoise.OUActionNoise import OUActionNoise
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

# TODO -> test 0.01 tau, critic learning rate to 0.00005, implement ornstein, add qvalue and gradient monitoring
# TODO -> add l2 regularization, batch normalization, test 0.98 and 0.95 discount,  
# Sharp decline due to policy deterministic, weak exploration, instability in qval predictions
LOAD_MODELS = False

env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
# env = gym.make("Pendulum-v1", render_mode="human")

num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print("Actions: ", num_actions, "\n",
      "States: ", num_states, "\n")

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

actor_model = get_actor(num_states, upper_bound)
critic_model = get_critic(num_states, num_actions)

target_actor = get_actor(num_states, upper_bound)
target_critic = get_critic(num_states, num_actions)

# Making the weights equal initially
if LOAD_MODELS:
    actor_model.load_weights("./weights/biped_actor_2.weights.h5")
    critic_model.load_weights("./weights/biped_critic_2.weights.h5")
    target_actor.load_weights("./weights/actor.weights.h5")
    target_critic.load_weights("./weights/biped_target_critic.weights.h5")
    print("MODELS_LOADED, PROCEEDING TRAINING")
else:
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.0001
actor_lr = 0.00005
# Weight decay
weight_decay = 0.0001

critic_optimizer = keras.optimizers.Adam(critic_lr, weight_decay=weight_decay)
actor_optimizer = keras.optimizers.Adam(actor_lr)

total_episodes = 500
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
# For noise generation
sigma = 0.1
# OUActionNoise
ou_noise = OUActionNoise(num_actions, 10)

buffer = Buffer(buffer_capacity=1000000,
                batch_size=128,
                num_states=num_states,
                num_actions=num_actions)


def policy(state):
    sampled_actions = keras.ops.squeeze(actor_model(state))
    noise = ou_noise.sample()
    # Adding noise to action
    legal_action = np.clip(sampled_actions + noise, lower_bound, upper_bound)
    return np.squeeze(legal_action)


critic_loss_history = []
actor_loss_history = []


def update(
    state_batch,
    action_batch,
    reward_batch,
    next_state_batch,
):
    # Training and updating Actor & Critic networks.
    # See Pseudo Code.
    with tf.GradientTape() as tape:
        target_actions = target_actor(next_state_batch, training=True)

        y = reward_batch + gamma * target_critic(
            keras.ops.concatenate((next_state_batch, target_actions), axis=1), training=True
        )

        critic_value = critic_model(
            keras.ops.concatenate((state_batch, action_batch), axis=1), training=True)
        critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))
        critic_loss_history.append(critic_loss)

    critic_grad = tape.gradient(
        critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, critic_model.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = actor_model(state_batch, training=True)
        critic_value = critic_model(keras.ops.concatenate((state_batch, actions), axis=1), training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -keras.ops.mean(critic_value)
        actor_loss_history.append(actor_loss)

    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor_model.trainable_variables)
    )

    # We compute the loss and update parameters


def learn():
    # Get sampling range
    record_range = min(buffer.buffer_counter, buffer.buffer_capacity)
    # Randomly sample indices
    batch_indices = np.random.choice(record_range, buffer.batch_size)

    # Convert to tensors
    state_batch = keras.ops.convert_to_tensor(
        buffer.state_buffer[batch_indices])
    action_batch = keras.ops.convert_to_tensor(
        buffer.action_buffer[batch_indices])
    reward_batch = keras.ops.convert_to_tensor(
        buffer.reward_buffer[batch_indices])
    reward_batch = keras.ops.cast(reward_batch, dtype="float32")
    next_state_batch = keras.ops.convert_to_tensor(
        buffer.next_state_buffer[batch_indices]
    )

    update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * \
            tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)


# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):
    prev_state, _ = env.reset()
    episodic_reward = 0

    while True:
        tf_prev_state = keras.ops.expand_dims(
            keras.ops.convert_to_tensor(prev_state), 0
        )

        action = policy(tf_prev_state)
        # Receive state and reward from environment.
        state, reward, terminated, truncated, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        learn()

        update_target(target_actor, actor_model, tau)
        update_target(target_critic, critic_model, tau)

        if terminated or truncated:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    print("Buffer size: ", buffer.buffer_counter)
    avg_reward_list.append(avg_reward)

# Save the weights
actor_model.save_weights("biped_actor_4.weights.h5")
critic_model.save_weights("biped_critic_4.weights.h5")

target_actor.save_weights("biped_target_actor_4.weights.h5")
target_critic.save_weights("biped_target_critic_4.weights.h5")

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
plt.plot(critic_loss_history)
plt.xlabel("Update call")
plt.ylabel("Critic loss")
plt.show()
plt.plot(actor_loss_history)
plt.xlabel("Update call")
plt.ylabel("Actor loss")
plt.show()
