import gym
import matplotlib.pyplot as plt
import random
import keras
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from FramePreprocess import FramePreprocess
from ExperienceReplay import ExperienceReplay
from DQN import DQN


# NOTES:
# How would multi frame states work with GANs?
# Can GANs produce exactly the same frames they are fed?

# split the frames on positive/other rewards
# no need for 4 dimension states for gans
# idea: only generate positive states from gans
# pytorch/keras gan tutorials cifr10
def main():
    #env = gym.make('ALE/Breakout-v5', obs_type = "grayscale", full_action_space=False)#, render_mode="human")
    #env = FramePreprocess(env)
    env = make_atari("BreakoutNoFrameskip-v4")
    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(42)
    #print(env.action_space)
    #print(env.observation_space.shape)
    #print(env.unwrapped.get_action_meanings())
    #help(env.unwrapped)
    exp_replay = ExperienceReplay()

    env.reset()
    
    # for _ in range(50):
    #     action = env.action_space.sample()
    #     frame, reward, done, _ = env.step(action)
        #if reward == 1: keep frame for GAN training
        #exp_replay.add_experience(frame, action, reward, done)


if __name__ == "__main__":
    main()