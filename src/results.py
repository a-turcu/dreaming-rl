import gym
import matplotlib.pyplot as plt
import numpy as np
from atari_wrappers import make_atari, wrap_deepmind
import cv2
from tensorflow import keras
import tensorflow as tf
import os
import sys


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
    model = keras.models.load_model('C:/Users/alexa/OneDrive/Desktop/model25000.h5')

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
    #path = "C:/Users/alexa/OneDrive/Desktop/train_GAN/ep16633_rr35.51.npy"
    path = "successful_frames_short.npy"
    successful_frames = np.load(path)
    #successful_frames.resize(successful_frames.shape[0], 28, 28, 1)
    #successful_frames = successful_frames / 127.5 - 1.
    #successful_frames = np.expand_dims(successful_frames, axis=3)
    #np.set_printoptions(threshold=sys.maxsize)
    for frame in successful_frames:
        cv2.imshow("Frame", frame)
        #print(frame)
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


def plot_gan_metrics():
    path = "C:/Users/alexa/Documents/RUG/Year 3/Bachelor Project/GAN_data/performance_graph/"
    name = {"d_loss": "d_losses.npy", "acc": "accs.npy", "g_loss": "g_losses.npy", "epoch": "epochs.npy"}
    d_losses = np.load(path + "d_losses.npy")
    accs = np.load(path + "accs.npy")
    g_losses = np.load(path + "g_losses.npy")
    epochs = np.load(path + "epochs.npy")
    #plt.plot(epochs, d_losses, label="Discriminator Loss")
    #plt.plot(epochs, g_losses, label="Generator Loss")
    plt.plot(epochs, accs, label="Accuracy")
    plt.show()

def main():
    
    #demo()
    #plot_rewards()
    show_successful_frames()
    #plot_gan_metrics()

if __name__ == "__main__":
    main()