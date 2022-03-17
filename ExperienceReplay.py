import numpy as np
import random

STATE_LEN = 4
BATCH_SIZE = 32

class ExperienceReplay(object):
    def __init__(self, capacity = 100000):
        self.capacity = capacity
        self.frames = np.empty((self.capacity, 84, 84), dtype=np.uint8)
        self.actions = np.empty(self.capacity, dtype=np.uint8)
        self.rewards = np.empty(self.capacity, dtype=np.uint16) # maybe dtype=np.float16?
        self.done = np.empty(self.capacity, dtype=np.bool)
        self.idx = 0
        self.count = 0

    def add_experience(self, frame, action, reward, done):
        """
        A state is made out of 4 (STATE_LEN) frames, so there is no need to save both 
        state(t) and state(t+1). The two states have 3 common frames. Both states can 
        be accessed from self.frames. 
        """
        self.frames[self.idx] = frame
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.count += 1

    def get_state(self, index):
        
        # !Often raises error
        if index >= STATE_LEN:
            state = self.frames[index-STATE_LEN+1 : index+1]
            return state
       # else:
        #    raise ValueError("Index is too small")

    def pick_batch_members(self):
        """
        Creates a random selection of valid observations to use in a minibatch.
        """
        
        indices = np.empty(BATCH_SIZE, dtype=np.uint8)
        
        for i in range(BATCH_SIZE):
            
            conditions = False
            while conditions == False:
                
                conditions = True
                candidate_idx = random.randint(STATE_LEN, self.count-1)

                # Don't pick the same index twice                
                if candidate_idx in indices:
                    conditions = False

                # It is not desirable to create a state out of frames that led to the end of an episode.
                if self.done[candidate_idx-STATE_LEN : candidate_idx].any() is True:
                    conditions = False

                # other conditions?

            indices[i] = candidate_idx
        
        return indices

    def sample_experiences(self):
        
        indices = self.pick_batch_members()
        #print(indices)
        states_t = np.empty((BATCH_SIZE, STATE_LEN, 84, 84), dtype=np.uint8)
        states_t1 = np.empty((BATCH_SIZE, STATE_LEN, 84, 84), dtype=np.uint8)
        actions = np.empty(BATCH_SIZE, dtype=np.uint8)
        rewards = np.empty(BATCH_SIZE, dtype=np.uint16)
        done = np.empty(BATCH_SIZE, dtype=np.bool)

        # Create the state(t) and state(t+1) lists out of overlapping frames
        for i, j in enumerate(indices):
            states_t[i] = self.get_state(j - 1)
            states_t1[i] = self.get_state(j)
            actions[i] = self.actions[j]
            rewards[i] = self.rewards[j]
            done[i] = self.done[j]
        
        # reshaping for GANs?
        return states_t, actions, rewards, states_t1, done
