import numpy as np


class Buffer:
    def __init__(self, capacity=300000, states=24, actions=4, batch_size=128):
        self.capacity = capacity
        self.batch_size = batch_size
        self.counter = 0

        self.state_mem = np.zeros((capacity, states), dtype=np.float32)
        self.action_mem = np.zeros((capacity, actions), dtype=np.float32)
        self.reward_mem = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state_mem = np.zeros((capacity, states), dtype=np.float32)
        self.done_mem = np.zeros((capacity, 1), dtype=np.float32)

    def save(self, state, action, reward, next, done):
        index = self.counter % self.capacity
        self.state_mem[index] = state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.next_state_mem[index] = next
        self.done_mem[index] = done
        self.counter += 1

    def get_batch(self):
        index = np.random.choice(min(self.counter, self.capacity),
                                 size=self.batch_size,
                                 replace=False)

        return self.state_mem[index], \
            self.action_mem[index], \
            self.reward_mem[index], \
            self.next_state_mem[index], \
            self.done_mem[index]
