from tensorflow.keras import layers
from tensorflow import keras

# maybe try 400 - 300 neuron setup?


def get_critic(num_states, num_actions):
    # State as input
    state_input = layers.Input(shape=(num_states + num_actions,))

    out = layers.Dense(400, activation="relu")(state_input)
    out = layers.Dense(300, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = keras.Model(state_input, outputs)

    return model


def get_actor(num_states, upper_bound):
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(400, activation="relu")(inputs)
    out = layers.Dense(300, activation="relu")(out)  # maybe throw this line out
    outputs = layers.Dense(4, activation="tanh",
                           kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    model = keras.Model(inputs, outputs)
    return model
