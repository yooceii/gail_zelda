import os
import sys
import gym
import gym_gvgai
from ZeldaEnv import ZeldaEnv
from model import GAIL
from dataset.dataset import ExpertDataset
from baselines.bench import Monitor
from baselines import logger
sys.path.append("/home/chang/gail/nnrunner/a2c_gvgai")
import nnrunner.a2c_gvgai.env as gvgai_env

# Load the expert dataset
dataset = ExpertDataset(expert_path='dataset/expert_zelda.npz', traj_limitation=-1, verbose=1)
# env = gvgai_env.make_gvgai_env("Pendulum-v0", 2, 571846)
# env = gvgai_env.make_gvgai_env("gvgai-zelda-lvl0-v0", 2, 571846)
env = Monitor(gym.make("gvgai-zelda-lvl0-v0"), logger.get_dir() and os.path.join(logger.get_dir(), "monitor.json"))
env = ZeldaEnv(env)
env.seed(571846) 
level = "/home/chang/gail/levels/1_level.txt"
env.unwrapped._setLevel(level)
try:
    # model = GAIL("CnnPolicy", 'Pendulum-v0', dataset, verbose=1)
    model = GAIL("CnnPolicy", env, dataset, verbose=1, tensorboard_log="./gail_log")

    model.pretrain(dataset, n_epochs=1e4)
    model.save("bc_zelda")
    # Note: in practice, you need to train for 1M steps to have a working policy
    model.learn(total_timesteps=100000000)
    model.save("gail_zelda")

finally:
    pass

del model # remove to demonstrate saving and loading

# model = GAIL.load("gail_pendulum")

# env = gym.make('Pendulum-v0')
# obs = env.reset()
# while True:
#   action, _states = model.predict(obs)
#   obs, rewards, dones, info = env.step(action)
#   env.render()
