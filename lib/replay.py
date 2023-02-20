import numpy as np

class ReplayMemory(object):
    def __init__(self, mem_depth, n_states, n_actions, 
                 state_dtype=None, action_dtype=None):
        # Initialize replay memory variables
        self.depth = mem_depth
        self.experience_counter = 0

        # Initialize memories data type
        if state_dtype is None:
            if (n_states <= (2**8-1)):
                self.state_dtype = np.uint8
            elif (n_states <= (2**16-1)):
                self.state_dtype = np.uint16
            elif (n_states <= (2**32-1)):
                self.state_dtype = np.uint32
            else:
                self.state_dtype = np.uint64
        if action_dtype is None:
            if (n_actions <= (2**8-1)):
                self.action_dtype = np.uint8
            else:
                # Number of action exceed 255
                self.action_dtype = np.uint16
        
        # Experience Memories
        self.state_memory = np.zeros(self.depth, dtype=self.state_dtype)
        self.next_state_memory = np.zeros(self.depth, dtype=self.state_dtype)
        self.action_memory = np.zeros(self.depth, dtype=self.action_dtype)
        self.reward_memory = np.zeros(self.depth, dtype=np.float32)
        self.terminal_memory = np.zeros(self.depth, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.experience_counter % self.depth
        self.state_memory[index] = state
        self.next_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.experience_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.experience_counter, self.depth)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminal
