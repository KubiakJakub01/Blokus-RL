"""Module with trainer for class."""
import os
from pathlib import Path
from typing import Any

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.hparams import HParams
from src.utils import LOG_INFO, make_env
from src.ppo.agent import Agent
from src.ppo.memory import Memory


class Trainer:
    """Trainer for proximal policy optimization."""

    def __init__(self, hparams: HParams, device: torch.device):
        """Initialize the trainer."""
        self.hparams = hparams
        self.device = device
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(i, hparams) for i in range(hparams.num_envs)]
        )
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"

        # Init variables
        self.global_step = 0

        # Init model, optimizer and memory
        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=hparams.lr, eps=hparams.eps
        )
        self.memory = Memory(hparams, self.envs, self.device)

        # Init tensorboard
        self.writer = SummaryWriter(log_dir=f"{hparams.log_dir}/{hparams.run_name}")

    def train(self):
        """Train the model."""
        self.next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        self.next_done = torch.zeros(self.hparams.num_envs).to(self.device)

        for update in range(1, self.hparams.num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.hparams.anneal_lr:
                self.optimizer.param_groups[0]["lr"] = self._compute_anneal_lr(update)

            # Perform a training step
            self.next_obs, self.next_done = self._play_env()

            # Compute advantages and returns
            next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            if self.hparams.gae:
                advantages = self._compute_gae(next_value, self.next_done)
                returns = advantages + self.memory.values
            else:
                returns = self._compute_returns(next_value, self.next_done)
                advantages = returns - self.memory.values

            # flatten the batch
            b_obs = self.memory.obs.reshape(
                (-1,) + self.envs.single_observation_space.shape
            )
            b_logprobs = self.memory.logprobs.reshape(-1)
            b_actions = self.memory.actions.reshape(
                (-1,) + self.envs.single_action_space.shape
            )
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.memory.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.hparams.batch_size)
            clipfracs = []
            for epoch in range(self.hparams.num_epochs):
                np.random.shuffle(b_inds)
                # For each minibatch
                for start in range(
                    0, self.hparams.batch_size, self.hparams.minibatch_size
                ):
                    end = start + self.hparams.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.hparams.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.hparams.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss = self._calculate_policy_loss(mb_advantages, ratio)

                    # Value loss
                    v_loss = self._calculate_value_loss(
                        newvalue,
                        mb_inds,
                        b_returns,
                        b_values,
                        entropy,
                        pg_loss.detach(),
                    )

                    # Backpropagate
                    self.optimizer.zero_grad()

    def _play_env(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Play the environment for a number of steps."""
        for step in range(self.hparams.num_steps):
            self.global_step += 1 * self.hparams.num_envs
            self.memory.obs[step] = self.next_obs
            self.memory.dones[step] = self.next_done

            # Get action and value from the agent
            action, logproba, entropy, value = self._get_action_and_value(
                self.memory.obs[step]
            )
            self.memory.values[step] = value.flatten()
            self.memory.actions[step] = action
            self.memory.logprobs[step] = logproba

            # Step the environment
            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            self.memory.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(self.device),
                torch.Tensor(done).to(self.device),
            )

            # Log to tensorboard
            self._log_to_tb(self.global_step, info)

        return next_obs, next_done

    def _calculate_policy_loss(
        self, mb_advantages: torch.Tensor, ratio: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the policy loss.

        Policy objective is to maximize the following:
        `L = -E[min(ratio * A, clip(ratio, 1 - clip_coef, 1 + clip_coef) * A)]`

        Args:
            mb_advantages: Advantage tensor.
            ratio: Ratio tensor.

        Returns:
            Policy loss."""
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(
            ratio, 1 - self.hparams.clip_coef, 1 + self.hparams.clip_coef
        )
        return torch.max(pg_loss1, pg_loss2).mean()

    def _calculate_value_loss(
        self,
        newvalue: torch.Tensor,
        mb_inds: torch.Tensor,
        b_returns: torch.Tensor,
        b_values: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the value loss.

        Value objective is to minimize the following:
        `L = E[(V(s) - R)^2]`

        Args:
            newvalue: New value tensor.
            mb_inds: Minibatch indices.
            b_returns: Returns tensor.
            b_values: Values tensor.
            entropy: Entropy tensor.
            pg_loss: Policy gradient loss.
        
        Returns:
            Value loss."""
        newvalue = newvalue.view(-1)
        if self.hparams.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(
                newvalue - b_values[mb_inds],
                -self.hparams.clip_coef,
                self.hparams.clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        return v_loss

    def _compute_total_loss(
        self, pg_loss: torch.Tensor, entropy: torch.Tensor, v_loss: torch.Tensor
    ) -> torch.Tensor:
        """Compute the total loss.

        Total loss is computed as:
        `L = pg_loss - ent_coef * entropy + v_loss * vf_coef`

        Args:
            pg_loss: Policy gradient loss.
            entropy: Entropy tensor.
            v_loss: Value loss.

        Returns:
            Total loss."""
        entropy_loss = entropy.mean()
        loss = (
            pg_loss
            - self.hparams.ent_coef * entropy_loss
            + v_loss * self.hparams.vf_coef
        )
        return loss

    def _compute_gae(
        self, next_value: torch.Tensor, next_done: torch.Tensor
    ) -> torch.Tensor:
        advantages = torch.zeros_like(self.memory.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.hparams.num_steps)):
            if t == self.hparams.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.memory.dones[t + 1]
                nextvalues = self.memory.values[t + 1]
            delta = (
                self.memory.rewards[t]
                + self.hparams.gamma * nextvalues * nextnonterminal
                - self.memory.values[t]
            )
            advantages[t] = lastgaelam = (
                delta
                + self.hparams.gamma
                * self.hparams.gae_lambda
                * nextnonterminal
                * lastgaelam
            )
        return advantages

    @torch.inference_mode()
    def _compute_returns(
        self, next_value: torch.Tensor, next_done: torch.Tensor
    ) -> torch.Tensor:
        """Compute the returns.

        Returns are computed as:
        `R_t = r_t + gamma * (1 - done_{t+1}) * V(s_{t+1})`
        
        Args:
            next_value: Value of the next state.
            next_done: Whether the next state is done.
        
        Returns:
            Returns."""
        returns = torch.zeros_like(self.memory.rewards).to(self.device)
        for t in reversed(range(self.hparams.num_steps)):
            if t == self.hparams.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                next_return = next_value
            else:
                nextnonterminal = 1.0 - self.memory.dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = (
                self.memory.rewards[t]
                + self.hparams.gamma * nextnonterminal * next_return
            )
        return returns

    def _compute_anneal_lr(self, update: int) -> float:
        """Compute the annealed learning rate.

        Anneals learning rate is computed as:
        `lr = lr_0 * (1 - (update - 1) / num_updates)`

        Args:
            update: Current update step.
        
        Returns:
            Annealed learning rate."""
        frac = 1.0 - (update - 1.0) / self.hparams.num_updates
        return frac * self.hparams.learning_rate

    @torch.inference_mode()
    def _get_action_and_value(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value from the agent.
        
        Args:
            obs: Observation tensor.
        
        Returns:
            Tuple of action, log probability, entropy and value."""
        action, logproba, entropy, value = self.agent.get_action_and_value(obs)
        return action, logproba, entropy, value

    def _log(self, update: int):
        """Log the training progress.
        
        Args:
            update: Current update step."""
        if update % self.hparams.log_interval == 0:
            LOG_INFO(
                f"Update: {update}\t"
                f"Mean reward: {self.memory.rewards.sum(dim=0).mean().item()}\t"
                f"Mean value: {self.memory.values.mean().item()}\t"
                f"Mean entropy: {self.memory.entropies.mean().item()}\t"
                f"Mean logproba: {self.memory.logprobas.mean().item()}\t"
            )

    def _log_to_tb(self, update: int, item: dict[str, float]):
        """Log to tensorboard.
        
        Args:
            update: Current update step.
            item: Dictionary containing the episode return and length."""
        if update % self.hparams.log_interval == 0:
            self.writer.add_scalar(
                "reward", self.memory.rewards.sum(dim=0).mean().item(), self.global_step
            )
            self.writer.add_scalar(
                "value", self.memory.values.mean().item(), self.global_step
            )
            self.writer.add_scalar(
                "entropy", self.memory.entropies.mean().item(), self.global_step
            )
            self.writer.add_scalar(
                "logproba", self.memory.logprobas.mean().item(), self.global_step
            )
            self.writer.add_scalar(
                "episodic_return", item["episode"]["r"], self.global_step
            )
            self.writer.add_scalar(
                "episodic_length", item["episode"]["l"], self.global_step
            )
