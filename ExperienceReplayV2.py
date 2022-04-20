import numpy as np
import tensorflow as tf

BATCH_SIZE = 32

class ExperienceReplay(object):
    def __init__(self, capacity = 10000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.states_next = []
        self.dones = []

    def add_experience(self, state, action, reward, state_next, done):
        
        # make space for new experience
        if len(self.rewards) > self.capacity:
            del self.states[:1]
            del self.actions[:1]
            del self.rewards[:1]
            del self.states_next[:1]
            del self.dones[:1]

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_next.append(state_next)
        self.dones.append(done)
    
    def sample_experiences(self):
        
        indices = np.random.choice(range(len(self.rewards)), size=BATCH_SIZE)

        states_sample = np.array([self.states[i] for i in indices])
        actions_sample = [self.actions[i] for i in indices]
        rewards_sample = [self.rewards[i] for i in indices]
        states_next_sample = np.array([self.states_next[i] for i in indices])
        dones_sample = tf.convert_to_tensor([float(self.dones[i]) for i in indices])

        return states_sample, actions_sample, rewards_sample, states_next_sample, dones_sample
        
