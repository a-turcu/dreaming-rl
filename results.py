import gym
import matplotlib.pyplot as plt
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind
import cv2
from tensorflow import keras
import tensorflow as tf
import os


# Plot clipped running reward over episodes
def plot_rewards():
    path = "C:/Users/alexa/OneDrive/Desktop/graph_data/"
    rewards = np.load(path + "rewards.npy")
    episodes = np.load(path + "epsiodes.npy")
    plt.plot(episodes, rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Average Running Reward (100 episodes)")
    plt.title("Vanilla DQN")
    plt.show()
    #plt.savefig(path + "rewards.png")


# Demo of the best model yet
def demo():
    env = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True, scale=True, clip_rewards=False)
    env.seed(42)
    model = keras.models.load_model('C:/Users/alexa/OneDrive/Desktop/models/model20000.h5')

    while True:
        state = np.array(env.reset())
        episode_reward = 0
        for _ in range(10000):
            env.render()
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
            state_next, reward, done, _ = env.step(action)
            state_next = np.array(state_next)
            episode_reward += reward
            state = state_next
            if done:
                break
        print(f"Episode reward:{episode_reward}")


# Show the frames that led to a positive reward
def show_successful_frames():
    path = "C:/Users/alexa/OneDrive/Desktop/train_GAN/ep16633_rr35.51.npy"
    successful_frames = np.load(path)
    for frame in successful_frames:
        # Resize for printing
        resized = cv2.resize(frame, (400,400), interpolation = cv2.INTER_AREA)
        cv2.imshow("Frame", resized)
        cv2.waitKey(0)

# Join all successful frames for GAN training
def join_frames():
    all_successful_frames = np.empty((0, 84, 84, 4))
    path = "C:/Users/alexa/OneDrive/Desktop/train_GAN"
    file_names = os.listdir(path)
    
    for file_name in file_names:
        data = np.load(path + "/" + file_name)
        all_successful_frames = np.append(all_successful_frames, data, axis=0)
    
    print(np.shape(all_successful_frames))
    np.save("C:/Users/alexa/OneDrive/Desktop/all_successful_frames.npy", all_successful_frames)


def main():
    
    demo()
    #plot_rewards()
    #show_successful_frames()


if __name__ == "__main__":
    main()