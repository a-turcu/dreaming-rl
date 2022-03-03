import gym
import matplotlib.pyplot as plt

env = gym.make('ALE/Breakout-v5')
print(env.action_space.n)
print(env.observation_space.shape)
env.reset()
for _ in range(50):
    obs, _, _, _ = env.step(env.action_space.sample())


plt.title("Game image")
plt.imshow(env.render("rgb_array"))
plt.show()

# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()