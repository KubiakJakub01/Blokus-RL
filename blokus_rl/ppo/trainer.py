"""Module with trainer for class."""
import time
from collections import defaultdict
from typing import Any
from statistics import mean

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .agent import Agent
from .memory import Memory
from ..hparams import HParams
from ..utils import LOG_INFO, LOG_DEBUG, LOG_WARNING, make_envs


class Trainer:
    def __init__(self, hparams: HParams):
        """Initialize the trainer.
        
        Args:
            hparams: Hyperparameters for the trainer."""
        self.hparams = hparams
        self.writer = SummaryWriter(log_dir=self.hparams.log_dir)
        self.device = (
            "cuda" if torch.cuda.is_available() and self.hparams.cuda else "cpu"
        )
        self.envs = gym.vector.SyncVectorEnv(
            [make_envs(i, self.hparams) for i in range(self.hparams.num_envs)]
        )
        self.agent = Agent(self.envs, self.hparams).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.eps
        )
        self.memory = Memory(self.hparams, self.envs, self.device)
        self.running_vals = self._reset_running_vals()

        self._total_episodes = 0
        self._total_episodes_reward = 0

        LOG_INFO("Trainer initialized with device: %s", self.device)

    def train(self):
        """Train the agent."""

        # Set the initial values
        self.global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.hparams.num_envs).to(self.device)

        for update in range(1, self.hparams.num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.hparams.anneal_lr:
                self.optimizer.param_groups[0]["lr"] = self._compute_anneal_lr(update)

            # Play the environment for a number of s teps
            next_obs, next_done = self._play_env(next_obs, next_done)

            # bootstrap value if not done
            with torch.inference_mode():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                self.memory.advantages = self._compute_gae(next_value, next_done)
                self.memory.returns = self.memory.advantages + self.memory.values

            # flatten the batch
            batch = self.memory.get_flatten_batch()

            # Optimizing the policy and value network
            self._optimize_agent(batch)

            # Update the running values for the charts
            self._update_running_vals(
                {"SPS": int(self.global_step / (time.time() - start_time))},
                prefix="charts",
            )

            # Log the running values
            self._log(update)

            # Save the model
            if update % self.hparams.save_interval == 0:
                self._save_checpoint()

        # Save the final model
        self._save_checpoint()

        # Close the environment
        self.envs.close()
        self.writer.close()

    def _compute_anneal_lr(self, update: int) -> float:
        """Compute the annealed learning rate.

        Anneals learning rate is computed as:
        `lr = lr_0 * (1 - (update - 1) / num_updates)`

        hparams:
            update: Current update step.
        
        Returns:
            Annealed learning rate."""
        frac = 1.0 - (update - 1.0) / self.hparams.num_updates
        return frac * self.hparams.learning_rate

    def _play_env(self, next_obs: torch.Tensor, next_done: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Play the environment for a number of steps.
        
        Args:
            next_obs: Next observation.
            next_done: Whether the next state is done.
        
        Returns:
            Next observation and whether the next state is done."""
        for step in range(self.hparams.num_steps):
            self.global_step += 1 * self.hparams.num_envs
            self.memory.obs[step] = next_obs
            self.memory.dones[step] = next_done

            # Get action and value from the agent
            with torch.inference_mode():
                possible_moves = self.envs.get_attr("ai_possible_indexes")
                action, logproba, _, value = self.agent.get_action_and_value(
                    next_obs, possible_moves=possible_moves
                )
                self.memory.values[step] = value.flatten()
            self.memory.actions[step] = action
            self.memory.logprobs[step] = logproba

            # Step the environment
            next_obs, reward, terminated, _, info = self.envs.step(action.cpu().numpy())
            self.memory.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(self.device),
                torch.Tensor(terminated).to(self.device),
            )

            for item in info.get("final_info", []):
                if item is not None and "episode" in item:
                    self._total_episodes += 1
                    self._total_episodes_reward += item["episode"]["r"]
                    self._update_running_vals(
                        {
                            "episode_return": item["episode"]["r"],
                            "episode_length": item["episode"]["l"],
                        },
                        prefix="charts",
                    )
                    break

        return next_obs, next_done

    def _compute_gae(self, next_value, next_done) -> torch.Tensor:
        """Compute the generalized advantage estimation.

        GAE is computed as:
        `delta = r + gamma * V(s') * (1 - done) - V(s)`
        `advantage = delta + gamma * lambda * (1 - done) * advantage`

        Args:
            next_value: Value of the next state.
            next_done: Whether the next state is done.

        Returns:
            Generalized advantage estimation."""
        LOG_DEBUG(f'step: %d rewards: %s', self.global_step, self.memory.rewards)
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

    def _optimize_agent(self, batch: dict[str, torch.Tensor]):
        b_inds = np.arange(self.hparams.batch_size)
        clipfracs = []
        for epoch in range(self.hparams.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.hparams.batch_size, self.hparams.minibatch_size):
                end = start + self.hparams.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    batch["obs"][mb_inds], batch["actions"].long()[mb_inds]
                )
                logratio = newlogprob - batch["logprobs"][mb_inds]
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

                mb_advantages = batch["advantages"][mb_inds]
                if self.hparams.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.hparams.clip_coef, 1 + self.hparams.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.hparams.clip_vloss:
                    v_loss_unclipped = (newvalue - batch["returns"][mb_inds]) ** 2
                    v_clipped = batch["values"][mb_inds] + torch.clamp(
                        newvalue - batch["values"][mb_inds],
                        -self.hparams.clip_coef,
                        self.hparams.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - batch["returns"][mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - batch["returns"][mb_inds]) ** 2).mean()

                # Total loss
                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - self.hparams.ent_coef * entropy_loss
                    + v_loss * self.hparams.vf_coef
                )

                # Backpropagate and update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.hparams.max_grad_norm
                )
                self.optimizer.step()

            if self.hparams.target_kl is not None:
                if approx_kl > self.hparams.target_kl:
                    break

        y_pred, y_true = batch["values"].cpu().numpy(), batch["returns"].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log the losses
        self._update_running_vals(
            {"learning_rate": self.optimizer.param_groups[0]["lr"]}, prefix="charts"
        )
        self._update_running_vals(
            {
                "loss": loss.item(),
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": np.mean(clipfracs),
                "explained_variance": explained_var,
            },
            prefix="losses",
        )

    def _reset_running_vals(self):
        """Reset the running values."""
        return defaultdict(list)

    def _update_running_vals(self, items: dict[str, Any], prefix=None):
        """Update the running values.
        
        Args:
            items: Dictionary containing the values.
            prefix: Prefix for the keys."""
        for key, value in items.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.item()
            if prefix is None:
                self.running_vals[key].append(value)
            else:
                self.running_vals[f"{prefix}/{key}"].append(value)

    def _log(self, update: int):
        """Log the running values."""
        if self.hparams.logging:
            if update % self.hparams.log_interval == 0:
                LOG_INFO(
                    "global_step: %d | episodic_return: %.2f | loss: %.2f | SPS: %d",
                    self.global_step,
                    mean(self.running_vals["charts/episode_return"])
                    if len(self.running_vals["charts/episode_return"]) > 0
                    else 0,
                    mean(self.running_vals["losses/loss"]),
                    mean(self.running_vals["charts/SPS"]),
                )

            # Log the running values
            if update % self.hparams.tb_log_interval == 0:
                for key, value in self.running_vals.items():
                    self.writer.add_scalar(key, mean(value), self.global_step)
                self.running_vals = self._reset_running_vals()
        else:
            self.running_vals = self._reset_running_vals()

    def _save_checpoint(self):
        """Save the model."""
        torch.save(
            self.agent.state_dict(),
            self.hparams.log_dir / f"model_{self.global_step}.pt",
        )

    @property
    def mean_episode_reward(self) -> float:
        """Mean episode reward."""
        if self._total_episodes == 0:
            LOG_WARNING("No episodes were played.")
            return 0
        return self._total_episodes_reward / self._total_episodes
