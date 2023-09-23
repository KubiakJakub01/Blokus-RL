import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from ..hparams import HParams
from ..utils import LOG_INFO


class SBTrainer:
    def __init__(self, hparams: HParams):
        self.hparams = hparams
        # Define device
        self.device = "cpu"
        # Create environment
        self.env = gym.make(self.hparams.gym_env)
        # Instantiate the agent
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.999,
            gae_lambda=0.98,
            ent_coef=0.01,
            verbose=1,
        )
        LOG_INFO("Initialized SBTrainer with device: %s" % self.device) 

    def train(self):
        self.model.learn(total_timesteps=self.hparams.total_timesteps)
        self.model.save(self.hparams.checkpoint_dir)
