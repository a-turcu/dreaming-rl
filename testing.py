import gym
import matplotlib.pyplot as plt
import random
import keras
import numpy as np
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from FramePreprocess import FramePreprocess
from ExperienceReplay import ExperienceReplay


def main():
  
    env = make_atari("BreakoutNoFrameskip-v4")
    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(42)

    data = [1,2,3]
    np.save("test.npy", np.array(data))
    data = np.load("test.npy")
    data = np.append(data, 4)
    np.save("test.npy", np.array(data))
    data = np.load("test.npy")
    print(data)
    # path = "C:/Users/alexa/Documents/RUG/Year 3/Bachelor Project/train_GAN/successful_frames2776.npy"
    # data = np.load(path)
    # for d in data:
    #     plt.imshow(d)
    #     plt.show()
    

if __name__ == "__main__":
    main()