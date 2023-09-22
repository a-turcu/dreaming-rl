from collections import deque
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32

class ExperienceReplay(object):
    def __init__(self, capacity = 10000):
        self.capacity = capacity
        self.states = deque([])
        self.actions = deque([])
        self.rewards = deque([])
        self.states_next = deque([])
        self.dones = deque([])

    def add_experience(self, state, action, reward, state_next, done):
        
        # make space for new experience
        # improve this
        # dequeue collections
        if len(self.rewards) > self.capacity:
            self.states.pop()
            self.actions.pop()
            self.rewards.pop()
            self.states_next.pop()
            self.dones.pop()

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
        
