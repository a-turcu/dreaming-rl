import cv2
from gym.core import ObservationWrapper
from gym.spaces import Box

class FramePreprocess(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.size = (84, 84)
        self.observation_space = Box(0.0, 1.0, (self.size[0], self.size[1], 1))

    # Other methods of preprocessing?
    def observation(self, frame):
        
        frame = frame[34:-16, :]
        
        frame = cv2.resize(frame, self.size)

        # divide by 255 for normalization (from [0, 255] to [0, 1])
        frame = frame.astype('float32') / 255

        return frame