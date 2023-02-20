from tf_agents.environments import suite_gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt 

class FrozenLakeEnv(object):
	def __init__(self, map_dim, frozen_prob=0.8):
		self.name = 'FrozenLake-v1'
		self.is_slippery = False
		self.map_dim = map_dim
		self.frozen_prob = frozen_prob
		self.map_desc = generate_random_map(size=self.map_dim, p=self.frozen_prob)
		self.kwargs = {'is_slippery': self.is_slippery, 'desc':self.map_desc}
		self.env = suite_gym.load(self.name, gym_kwargs=self.kwargs)

	def __call__(self):
		return self.env

	def print_spaces(self):
		print(self.env.observation_spec())
		print(self.env.action_spec())
	
	def display_map(self):
		self.env.reset()
		img = self.env.render(mode='rgb_array')
		plt.imshow(img)
		plt.axis('off')
		plt.show()