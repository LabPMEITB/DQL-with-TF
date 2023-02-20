"""
PROGRAMMER  : ANDI MUHAMMAD RIYADHUS ILMY
CREATE DATE : 2023/02/14 15:00
DESCRIPTION : Deep Q-Learning script with tensor flow.
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from lib.replay import ReplayMemory
from lib.qnetworks import QNetwork

class dql:
    def __init__(self,
                 environment,
                 learning_rate,
                 discount_factor,
                 epsilon,
                 eps_decay_rate,
                 eps_min_val,
                 total_episode,
                 maximum_timestep,
                 copy_target_interval,
                 experience_memory_size,
                 batch_size,
                 hidden_layers=None,
                 network_optimizer=None
                ):
        # Initialize Environment
        self.env = environment
        self.S = environment.observation_spec().maximum + 1
        self.A = environment.action_spec().maximum + 1
        self.action_space = [i for i in range(self.A)]

        # Initialize Learning Parameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.tau = eps_decay_rate
        self.eps_min = eps_min_val
        self.E = total_episode
        self.T = maximum_timestep
        self.C = copy_target_interval

        # Initialize Experience Replay
        self.EM = ReplayMemory(experience_memory_size, self.S, self.A)
        self.X = batch_size

        # Initialize Q-Network and Target Network
        self.QNetwork = QNetwork(self.A, h_layers=hidden_layers)
        self.TNetwork = QNetwork(self.A, h_layers=hidden_layers)
        ## Compile Networks
        if network_optimizer is None:
            optimizer = Adam(learning_rate=self.alpha)
        else:
            optimizer = network_optimizer
        self.QNetwork._name = "Q-Network"
        self.QNetwork.compile(optimizer=optimizer, loss='mse')
        self.TNetwork._name = "T-Network"
        self.TNetwork.compile(optimizer=optimizer, loss='mse')

        # Initialize Analitics
        self.epsilon_history = []

    def get_q(self, state, network, verbose=0):
        # Create One Hot Encoding Matrix for NN input
        input_matrix = [0 for i in range(self.S)]
        input_matrix[state] = 1
        input_array = np.array(input_matrix, ndmin=1)
        input_array = input_array[None,...]
        # Feed-forward network
        with tf.device('/GPU:0'):
            q_values = network.predict(input_array, verbose=verbose)
        return q_values

    def update_network(self, state, target_outputs, network, verbose=0):
        # Create One Hot Encoding Matrix for NN input
        input_matrix = [0 for i in range(self.S)]
        input_matrix[state] = 1
        input_array = np.array(input_matrix, ndmin=1)
        input_array = input_array[None,...]
        # Feed-forward network
        with tf.device('/GPU:0'):
            network.fit(input_array, target_outputs, verbose=verbose)
    
    def choose_action(self, state, network):
        # Calculate Random Value
        random_number = np.random.random()
        # Choose Action Policy
        if (random_number > self.epsilon):
            q_values = self.get_q(state, network)
            action = np.argmax(q_values)
        else:
            ## Choose Random Action
            action = np.random.choice(self.action_space)
        return action
    
    def evaluate(self):
        # Sample batch
        if (self.EM.experience_counter < self.X):
            return
        else:
            sampled_experiences = self.EM.sample_buffer(self.X)
            for i in range(self.X):
                # Sample data
                state = sampled_experiences[0][i]
                action = sampled_experiences[1][i]
                reward = sampled_experiences[2][i]
                next_state = sampled_experiences[3][i]
                is_terminal = sampled_experiences[4][i]

                # Get Q-values
                Q_values = self.get_q(state, self.QNetwork)
                Q_next = self.get_q(next_state, self.TNetwork)

                # Calculate target value
                Q_target = Q_values.copy()
                if is_terminal:
                    Q_target[0][action] = reward
                else:
                    Q_target[0][action] = reward + self.gamma*np.max(Q_next)

                # Update weights
                self.update_network(state, Q_target, self.QNetwork)

    def show_network_arch(self):
        plot_model(
            self.QNetwork.build_graph(self.S),
            to_file='Q-Network.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=True
        )
        plot_model(
            self.TNetwork.build_graph(self.S),
            to_file='T-Network.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=True
        )

    def copy_weight_to_target(self):
        return self.TNetwork.set_weights(self.QNetwork.get_weights())

    def decrease_epsilon(self):
        if self.epsilon > self.eps_min :
            self.epsilon -= self.tau
        else :
            self.epsilon = self.eps_min


    def learn(self):
        # Initialize Q-Network
        self.QNetwork.build_graph(self.S, name="Q-Network").summary()
        self.TNetwork.build_graph(self.S, name="T-Network").summary()
        
        # Copy weight
        # print(self.get_q(1, self.QNetwork))
        # print(self.get_q(1, self.TNetwork))
        self.copy_weight_to_target()
        ## Check
        # print(self.get_q(1, self.QNetwork))
        # print(self.get_q(1, self.TNetwork))

        # self.QNetwork.summary()
        # self.TNetwork.summary()

        # Learn for a number of episodes
        start = time.time()
        for e in range(self.E):
            # Reset Environment, move agent to the start state
            time_step = self.env.reset()
            st = time_step.observation
            
            # Print every 10% progress
            if not (e % 10):
                print(f"\nEpisode {e} [{time.time()-start}s]", end="")

            # Update Epsilon
            if e:
               self.decrease_epsilon() 

            self.epsilon_history.append(self.epsilon)

            # Perform an episode for T steps or until the agent reaches the goal
            t = 0
            while (not time_step.is_last()) or (t == self.T):
                # Perform Action
                at = self.choose_action(st, self.QNetwork)

                # Observe Environment
                time_step = self.env.step(at)
                rt = time_step.reward
                st1 = time_step.observation

                # Store Experience
                self.EM.store_transition(st, at, rt, st1, time_step.is_last())

                # Update
                self.evaluate()

                # Copy weight if interval reached
                if t % self.C == 0:
                    print('.', end='')
                    self.copy_weight_to_target()

                # Update state
                st = st1

                # Increase time step
                t +=1
        
        print("\nExecution time = {0}s".format(time.time()-start))