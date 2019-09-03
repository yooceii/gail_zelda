import gym
import numpy as np

class ZeldaEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            if info.get("winner") == 'PLAYER_WINS':
                info["episode"]['c'] = 1
            else:
                info["episode"]['c'] = 0
        return obs, reward, done, info
