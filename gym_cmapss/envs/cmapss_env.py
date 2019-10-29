import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class CMAPSSEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    DATA_DIR = 'data/observations.npy'

    def __init__(self):
        self.reward_range = (-50.0, 50.0)
        self._create_env()
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.getObservationSize()))
        self.action_space = spaces.Discrete(2)
    
    def _create_env(self):
        self.cur_idx = 0
        self.observations = np.load('/home/mohan/pick-orient-place/yannik/gym-cmapss/data/observations.npy')

    def getObservationSize(self):
        return len(self.observations[0]), 1

    def step(self, action):
        self.cur_idx += 1
        cur_obs = self.observations[self.cur_idx]

        if action == 0: # no repair, next observation
            if self.cur_idx == len(self.observations)-1:
                cur_reward = -50 # engine failed
                done = True
                self.reset() # reset the environment when engine fails
            else:
                cur_reward = 50 # engine did not fail
                done = False

            return (cur_obs, cur_reward, done, None) # (observation, reward, done, info)
        
        else: # repair action
            done = True
            if self.cur_idx != len(self.observations)-1:
                self.reset()
                return (cur_obs, 0, done, None)
            else:
                self.reset()
                return (cur_obs, 50, done, None)

    
    def reset(self):
        self.cur_idx = 0

    def render(self, mode='human'):
        '''
        Don't do anything. Rendering is not required.
        '''
        pass

    def close(self):
        pass