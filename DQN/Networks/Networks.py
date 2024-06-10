import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(self, states, actions, l1=512, l2=256):
        super().__init__()

        self.dense_1 = tf.keras.layers.Dense(l1, "relu", name="Dense1")
        self.dense_2 = tf.keras.layers.Dense(l2, "relu", name="Dense2")
        self.dense_3 = tf.keras.layers.Dense(actions, "relu", name="Dense3")

    def call(self, inputs):
        dense_1_out = self.dense_1(inputs)
        dense_2_out = self.dense_2(dense_1_out)
        out = self.dense_3(dense_2_out)
        return out
