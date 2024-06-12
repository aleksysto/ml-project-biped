import tensorflow as tf
from Buffer.Buffer import Buffer
from Networks.Networks import Actor, Critic
from OUActionNoise.OUActionNoise import OUActionNoise


class Agent:

    def __init__(self,
                 states=24,
                 actions=4,
                 batch_size=128,
                 actor_lr=0.0005,
                 critic_lr=0.0005,
                 gamma=0.99,
                 tau=0.005,
                 std_dev=0.1,
                 c=0.3,
                 memory_len=300000):
        self.states = states
        self.actions = actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.std_dev = std_dev
        self.c = c
        self.noise_generator = OUActionNoise(actions, 10)

        self.memory = Buffer(memory_len, states, actions, batch_size)

        self.actor = Actor(states, actions)
        self.actor_target = Actor(states, actions)

        self.critic = Critic(states, actions)
        self.critic_target = Critic(states, actions)

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        self.loss_fn = tf.keras.losses.Huber()
        self.optimizer_a = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.optimizer_c = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    @tf.function
    def noise_action(self, state):
        noise = self.noise_generator.sample()
        action = self.actor(state[None, :]) + noise
        return action[0]

    @tf.function
    def greedy_action(self, state):
        action = self.actor(state[None, :])
        return action[0]

    def save(self, state, action, reward, next, done):
        self.memory.save(state, action, reward, next, done)

    def replay(self):
        if self.memory.counter < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.get_batch()
        self.train_critic(states, actions, rewards, next_states, dones)
        self.train_actor(states)
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        noise = tf.random.normal(
            shape=(1, self.actions), mean=0.0, stddev=self.std_dev)
        next_actions = self.actor(next_states) + \
            tf.clip_by_value(noise, -1.0 * self.c, self.c)

        next_q = self.critic_target([next_states, next_actions])
        y = rewards + (1 - dones) * self.gamma \
            * next_q

        with tf.GradientTape() as tape:
            q_1 = self.critic([states, actions])
            loss = self.loss_fn(y, q_1)
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        self.optimizer_c.apply_gradients(
            zip(gradients, self.critic.trainable_variables))

    @tf.function
    def train_actor(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q = self.critic([states, actions])
            loss = -tf.reduce_mean(q)
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer_a.apply_gradients(
            zip(gradients, self.actor.trainable_variables))

    def soft_update(self, evaluate_net, target_net):
        evaluate_weight = evaluate_net.get_weights()
        target_weight = target_net.get_weights()
        for i in range(len(evaluate_weight)):
            target_weight[i] = self.tau * evaluate_weight[i] + \
                (1 - self.tau) * target_weight[i]
        target_net.set_weights(target_weight)
