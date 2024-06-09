import tensorflow as tf
import keras.ops as ops


class Actor(tf.keras.Model):

    def __init__(self, states=24, actions=4, l1=400, l2=300):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(l1, "relu", name="Dense1")
        self.dense_2 = tf.keras.layers.Dense(l2, "relu", name="Dense2")
        self.dense_3 = tf.keras.layers.Dense(actions, "tanh", name="Dense3")

        # build and call network
        self.build((None, states))
        inputs = tf.keras.Input(shape=(states,))
        self.call(inputs)

    def call(self, inputs):
        dense_1_out = self.dense_1(inputs)
        dense_2_out = self.dense_2(dense_1_out)
        out = self.dense_3(dense_2_out)
        return out


class Critic(tf.keras.Model):
    def __init__(self, states=24, actions=4, l1=400, l2=300):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(l1, "relu", name="Dense1")
        self.dense_2 = tf.keras.layers.Dense(l2, "relu", name="Dense2")
        self.dense_3 = tf.keras.layers.Dense(1, name="dense_3")

        # build and call
        self.build([(None, states), (None, actions)])
        state = tf.keras.Input(shape=(states,))
        action = tf.keras.Input(shape=(actions,))
        self.call([state, action])

    def call(self, inputs):
        state = inputs[0]
        action = inputs[1]

        concat = ops.concatenate([state, action], axis=1)
        dense_1_out = self.dense_1(concat)
        dense_2_out = self.dense_2(dense_1_out)
        output = self.dense_3(dense_2_out)
        return output
