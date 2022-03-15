import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import matplotlib.pyplot as plt
import cv2

class ObsPreprocess(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.size = (64, 64)
        self.observation_space = Box(0.0, 1.0, (self.size[0], self.size[1], 1))

    def observation(self, obs):
        
        obs = obs[34:-16, :]
        
        obs = cv2.resize(obs, (64, 64))
        return obs


env = gym.make('ALE/Breakout-v5', obs_type = "grayscale", full_action_space=False)#, render_mode="human")
env = ObsPreprocess(env)
# 1 useless action?
# print(env.action_space)
#print(env.observation_space.shape)
# print(env.unwrapped.get_action_meanings())
#help(env.unwrapped)

env.reset()
for _ in range(50):
    obs, _, _, _ = env.step(env.action_space.sample())
