import gym
import matplotlib.pyplot as plt
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind
import cv2

def main():
  
    env = make_atari("BreakoutNoFrameskip-v4")
    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(42)

    # state = env.reset()
    # for _ in range(100):
    #     action = env.action_space.sample()
    #     env.step(action)
    #     env.render()



    # rewards = np.load("C:/Users/alexa/OneDrive/Desktop/graph_data/rewards.npy")
    # episodes = np.load("C:/Users/alexa/OneDrive\Desktop\graph_data/epsiodes.npy")
    # plt.plot(rewards, episodes)
    # plt.plot(rewards, episodes, "or")
    # plt.show()
    path = "C:/Users/alexa/OneDrive/Desktop/train_GAN/successful_frames19936.npy"
    data = np.load(path)
    for d in data:
        cv2.imshow("breakout", d)
        cv2.waitKey(0)
    

if __name__ == "__main__":
    main()