import gym
import matplotlib.pyplot as plt
import random

from FramePreprocess import FramePreprocess
from ExperienceReplay import ExperienceReplay

def greedy(q_vals):
    """
    Picks the action with the largest Q value
    """

    max_val = max(q_vals)
    indices = [i for i, x in enumerate(q_vals) if x == max_val]

    return random.choice(indices)


def eps_greedy(q_vals):
    """
    Picks the action with the largest Q value with a probability of 1-epsilon
    """

    eps = 0.1

    if random.random() > eps:
        return greedy(q_vals)
    else:
        return random.randint(0, 2)


def main():
    env = gym.make('ALE/Breakout-v5', obs_type = "grayscale", full_action_space=False)#, render_mode="human")
    env = FramePreprocess(env)
    # 1 useless action?
    # print(env.action_space)
    #print(env.observation_space.shape)
    # print(env.unwrapped.get_action_meanings())
    #help(env.unwrapped)
    exp_replay = ExperienceReplay()

    env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        frame, reward, done, _ = env.step(action)
        exp_replay.add_experience(frame, action, reward, done)

    #st, a, r, st1, d = exp_replay.sample_experiences()

if __name__ == "__main__":
    main()