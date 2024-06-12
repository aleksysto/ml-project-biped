import tensorflow as tf
from Buffer.Buffer import Buffer
from Networks.Networks import Critic


class Agent:

    def __init__(self,
                 states=24,
                 actions=4,
                 batch_size=128,
                 lr=0.0005,
                 gamma=0.99,
                 tau=0.005,
                 delay_interval=8,
                 memory_len=300000):
        self.states = states
        self.actions = actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.delay_interval = delay_interval

        self.memory = Buffer(memory_len, states, actions, batch_size)

        self.critic = Critic(states, actions)
        self.critic_target = Critic(states, actions)

        self.critic_target.set_weights(self.critic.get_weights())

        self.loss_fn = tf.keras.losses.Huber()
        self.optimizer_c = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def action(self, state):
        action = self.critic(state[None, :])
        return action[0]

    def save(self, state, action, reward, next, done):
        self.memory.save(state, action, reward, next, done)

    def replay(self, time):
        if self.memory.counter < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.get_batch()
        self.train_critic(states, actions, rewards, next_states, dones)
        if self.memory.counter % self.delay_interval == 0:
            self.soft_update(self.critic, self.critic_target)

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        q_next = self.critic_target(next_states)
        y = rewards + (self.gamma * q_next) * (1 - dones)
        y = tf.reduce_max(y, axis=1)
        with tf.GradientTape() as tape:
            q = self.critic(states)
            q = tf.reduce_max(q, axis=1)
            loss = self.loss_fn(y, q)
        gradients = tape.gradient(loss, self.critic.trainable_variables)
        gradients = [tf.clip_by_value(gradient, -1, 1)
                     for gradient in gradients]
        self.optimizer_c.apply_gradients(
            zip(gradients, self.critic.trainable_variables))

    def soft_update(self, evaluate_net, target_net):
        evaluate_weight = evaluate_net.get_weights()
        target_weight = target_net.get_weights()
        for i in range(len(evaluate_weight)):
            target_weight[i] = self.tau * evaluate_weight[i] + \
                (1 - self.tau) * target_weight[i]
        target_net.set_weights(target_weight)
