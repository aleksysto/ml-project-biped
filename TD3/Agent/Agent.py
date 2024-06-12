import tensorflow as tf
from Buffer.Buffer import Buffer
from Networks.Networks import Actor, Critic


class Agent:

    def __init__(self,
                 states=24,
                 actions=4,
                 batch_size=128,
                 lr=0.0005,
                 critic_lr=0.0005,
                 actor_lr=0.0005,
                 gamma=0.99,
                 tau=0.005,
                 delay_interval=10,
                 std_dev=0.1,
                 c=0.3,
                 memory_len=300000):
        self.states = states
        self.actions = actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.delay_interval = delay_interval
        self.std_dev = std_dev
        self.c = c

        self.memory = Buffer(memory_len, states, actions, batch_size)

        self.actor = Actor(states, actions)
        self.actor_target = Actor(states, actions)

        self.critic_1 = Critic(states, actions)
        self.critic_1_target = Critic(states, actions)
        self.critic_2 = Critic(states, actions)
        self.critic_2_target = Critic(states, actions)

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())

        self.loss_fn = tf.keras.losses.Huber()
        self.optimizer_a = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.optimizer_c1 = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.optimizer_c2 = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    @tf.function
    def noise_action(self, state):
        noise = tf.random.normal(
            shape=(1, self.actions), mean=0.0, stddev=self.std_dev)
        action = self.actor(state[None, :]) + tf.clip_by_value(noise,
                                                               -1.0 * self.c,
                                                               self.c)
        return action[0]

    @tf.function
    def greedy_action(self, state):
        action = self.actor(state[None, :])
        return action[0]

    def save(self, state, action, reward, next, done):
        self.memory.save(state, action, reward, next, done)

    def replay(self, time):
        if self.memory.counter < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.get_batch()
        self.train_critic(states, actions, rewards, next_states, dones)
        if time % self.delay_interval == 0:
            self.train_actor(states)
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic_1, self.critic_1_target)
            self.soft_update(self.critic_2, self.critic_2_target)

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones):
        noise = tf.random.normal(
            shape=(1, self.actions), mean=0.0, stddev=self.std_dev)
        next_actions = self.actor(next_states) + \
            tf.clip_by_value(noise, -1.0 * self.c, self.c)

        next_q_1 = self.critic_1_target([next_states, next_actions])
        next_q_2 = self.critic_2_target([next_states, next_actions])
        y = rewards + (1 - dones) * self.gamma \
            * tf.math.minimum(next_q_1, next_q_2)

        with tf.GradientTape() as tape:
            q_1 = self.critic_1([states, actions])
            loss = self.loss_fn(y, q_1)
        gradients = tape.gradient(loss, self.critic_1.trainable_variables)
        self.optimizer_c1.apply_gradients(
            zip(gradients, self.critic_1.trainable_variables))

        # train critic network 2
        with tf.GradientTape() as tape:
            q_2 = self.critic_2([states, actions])
            loss = self.loss_fn(y, q_2)
        gradients = tape.gradient(loss, self.critic_2.trainable_variables)
        self.optimizer_c2.apply_gradients(
            zip(gradients, self.critic_2.trainable_variables))

    @tf.function
    def train_actor(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q = self.critic_1([states, actions])
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
