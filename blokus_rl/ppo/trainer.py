"""Module with trainer for class."""
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .agent import Agent
from .memory import Memory
from ..hparams import HParams
from ..utils import LOG_INFO, LOG_DEBUG, make_envs


class PPOTrainer:
    """Trainer for proximal policy optimization."""

    def __init__(self, hparams: HParams):
        """Initialize the trainer."""
        self.hparams = hparams
        self.device = (
            "cuda" if torch.cuda.is_available() and self.hparams.cuda else "cpu"
        )
        self.envs = gym.vector.SyncVectorEnv(
            [make_envs(i, hparams) for i in range(hparams.num_envs)]
        )
        assert isinstance(
            self.envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"

        # Init variables
        self.global_step = 0

        # Init model, optimizer and memory
        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.eps
        )
        self.memory = Memory(hparams, self.envs, self.device)

        # Init tensorboard
        self.writer = SummaryWriter(log_dir=hparams.log_dir)
        self.running_vals = self._reset_running_vals()

        # Log the hparams
        LOG_INFO("hparams: %s", self.hparams)
        self.hparams.save(self.hparams.log_dir / "hparams.yaml")

        LOG_INFO("Using device: %s", self.device)

    def train(self):
        """Train the model."""
        start_time = time.time()
        LOG_INFO(
            "Starting training at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        obs, _ = self.envs.reset(seed=self.hparams.seed)
        next_obs = torch.Tensor(obs).to(self.device)
        next_done = torch.zeros(self.hparams.num_envs).to(self.device)

        for update in range(1, self.hparams.num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.hparams.anneal_lr:
                self.optimizer.param_groups[0]["lr"] = self._compute_anneal_lr(update)

            # Perform a training step
            next_obs, next_done = self._play_env(next_obs, next_done)

            # Compute advantages and returns
            with torch.inference_mode():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = self._compute_gae(next_value, next_done)
                returns = advantages + self.memory.values

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
            for epoch in range(self.hparams.update_epochs):
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

                    with torch.inference_mode():
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
                        newvalue, mb_inds, b_returns, b_values
                    )

                    # Total loss
                    entropy_loss = entropy.mean()
                    loss = loss = (
                        pg_loss
                        - self.hparams.ent_coef * entropy_loss
                        + v_loss * self.hparams.vf_coef
                    )
                    # loss = self._compute_total_loss(
                    #     pg_loss.detach(), entropy_loss, v_loss.detach()
                    # )

                    # Backpropagate
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.hparams.max_grad_norm
                    )
                    self.optimizer.step()

                # Early stopping
                if self.hparams.target_kl is not None:
                    if approx_kl > self.hparams.target_kl:
                        break

            y_pred, y_true = (
                b_values.cpu().detach().numpy(),
                b_returns.cpu().detach().numpy(),
            )
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # Update the running values
            self._update_running_vals(
                {
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "SPS": int(self.global_step / (time.time() - start_time)),
                },
                prefix="charts",
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

            # Log the training progress
            # self._log()
            print("SPS:", int(self.global_step / (time.time() - start_time)))
            self._log_to_tb()

    def _play_env(self, next_obs, next_done) -> tuple[torch.Tensor, torch.Tensor]:
        """Play the environment for a number of steps."""
        for step in range(self.hparams.num_steps):
            self.global_step += 1 * self.hparams.num_envs
            self.memory.obs[step] = next_obs
            self.memory.dones[step] = next_done

            # Get action and value from the agent
            with torch.inference_mode():
                action, logproba, entropy, value = self.agent.get_action_and_value(
                    next_obs
                )
                self.memory.values[step] = value.flatten()
            self.memory.actions[step] = action
            self.memory.logprobs[step] = logproba

            # Step the environment
            next_obs, reward, terminated, _, info = self.envs.step(action.cpu().numpy())
            self.memory.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, self.next_done = (
                torch.Tensor(next_obs).to(self.device),
                torch.Tensor(terminated).to(self.device),
            )

            # Update the running values
            self._update_running_vals(
                {
                    "reward": reward.mean(),
                    "value": value.cpu().mean(),
                    "entropy": entropy.cpu().mean(),
                    "logproba": logproba.cpu().mean(),
                },
                prefix="env",
            )
            self._log_info(info)

        return next_obs, next_done

    def _calculate_policy_loss(
        self, mb_advantages: torch.Tensor, ratio: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the policy loss.

        Policy objective is to maximize the following:
        `L = -E[min(ratio * A, clip(ratio, 1 - clip_coef, 1 + clip_coef) * A)]`

        hparams:
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

        hparams:
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
        self, pg_loss: torch.Tensor, entropy_loss: torch.Tensor, v_loss: torch.Tensor
    ) -> torch.Tensor:
        """Compute the total loss.

        Total loss is computed as:
        `L = pg_loss - ent_coef * entropy + v_loss * vf_coef`

        hparams:
            pg_loss: Policy gradient loss.
            entropy_loss: Entropy loss.
            v_loss: Value loss.

        Returns:
            Total loss."""
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

    def _compute_returns(
        self, next_value: torch.Tensor, next_done: torch.Tensor
    ) -> torch.Tensor:
        """Compute the returns.

        Returns are computed as:
        `R_t = r_t + gamma * (1 - done_{t+1}) * V(s_{t+1})`
        
        hparams:
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

        hparams:
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
        
        hparams:
            obs: Observation tensor.
        
        Returns:
            Tuple of action, log probability, entropy and value."""
        action, logproba, entropy, value = self.agent.get_action_and_value(obs)
        return action, logproba, entropy, value

    def _log(self):
        """Log the training progress."""
        LOG_INFO(
            "global_step: %d | loss %.2f | value_loss %.2f | SPS %d",
            self.global_step,
            np.mean(self.running_vals["losses/loss"]),
            np.mean(self.running_vals["losses/value_loss"]),
            np.mean(self.running_vals["charts/SPS"]),
        )

    def _log_to_tb(self):
        """Log the running values to tensorboard."""
        for key, values in self.running_vals.items():
            self.writer.add_scalar(key, np.mean(values), self.global_step)
        self.running_vals = self._reset_running_vals()

    def _log_info(self, info: dict[str, Any]):
        for item in info.get("final_info", []):
            if item is not None and "episode" in item:
                print(
                    f"global_step={self.global_step}, episodic_return={item['episode']['r']}"
                )
                # LOG_DEBUG(
                #     "global_step: %d | episodic_return: %d | episodic_length: %d",
                #     self.global_step,
                #     item["episode"]["r"],
                #     item["episode"]["l"],
                # )
                self._update_running_vals(
                    {
                        "episodic_return": item["episode"]["r"],
                        "episodic_length": item["episode"]["l"],
                    },
                    prefix="charts",
                )
                break

    def _reset_running_vals(self):
        """Reset the running values."""
        return defaultdict(list)

    def _update_running_vals(self, items: dict[str, Any], prefix=None):
        """Update the running values.
        
        hparams:
            items: Dictionary containing the values.
            prefix: Prefix for the keys."""
        for key, value in items.items():
            if prefix is None:
                self.running_vals[key].append(value)
            else:
                self.running_vals[f"{prefix}/{key}"].append(value)


class Trainer:
    def __init__(self, hparams: HParams):
        self.hparams = hparams
        self.writer = SummaryWriter(log_dir=self.hparams.log_dir)
        self.device = (
            "cuda" if torch.cuda.is_available() and self.hparams.cuda else "cpu"
        )
        self.envs = gym.vector.SyncVectorEnv(
            [make_envs(i, self.hparams) for i in range(self.hparams.num_envs)]
        )
        self.agent = Agent(self.envs).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.eps
        )
        self.memory = Memory(self.hparams, self.envs, self.device)

    def train(self):
        # ALGO Logic: Storage setup
        # obs = torch.zeros(
        #     (self.hparams.num_steps, self.hparams.num_envs)
        #     + self.envs.single_observation_space.shape
        # ).to(self.device)
        # actions = torch.zeros(
        #     (self.hparams.num_steps, self.hparams.num_envs)
        #     + self.envs.single_action_space.shape
        # ).to(self.device)
        # logprobs = torch.zeros((self.hparams.num_steps, self.hparams.num_envs)).to(
        #     self.device
        # )
        # rewards = torch.zeros((self.hparams.num_steps, self.hparams.num_envs)).to(
        #     self.device
        # )
        # dones = torch.zeros((self.hparams.num_steps, self.hparams.num_envs)).to(
        #     self.device
        # )
        # values = torch.zeros((self.hparams.num_steps, self.hparams.num_envs)).to(
        #     self.device
        # )

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.hparams.num_envs).to(self.device)
        num_updates = self.hparams.total_timesteps // self.hparams.batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.hparams.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.hparams.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.hparams.num_steps):
                global_step += 1 * self.hparams.num_envs
                self.memory.obs[step] = next_obs
                self.memory.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    self.memory.values[step] = value.flatten()
                self.memory.actions[step] = action
                self.memory.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminated, _, info = self.envs.step(
                    action.cpu().numpy()
                )
                self.memory.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = (
                    torch.Tensor(next_obs).to(self.device),
                    torch.Tensor(terminated).to(self.device),
                )

                for item in info.get("final_info", []):
                    if item is not None and "episode" in item:
                        print(
                            f"global_step={global_step}, episodic_return={item['episode']['r']}"
                        )
                        self.writer.add_scalar(
                            "charts/episodic_return", item["episode"]["r"], global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", item["episode"]["l"], global_step
                        )
                        break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
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
                returns = advantages + self.memory.values

            # flatten the batch
            b_obs = self.memory.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.memory.logprobs.reshape(-1)
            b_actions = self.memory.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.memory.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.hparams.batch_size)
            clipfracs = []
            for epoch in range(self.hparams.update_epochs):
                np.random.shuffle(b_inds)
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
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.hparams.clip_coef, 1 + self.hparams.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
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

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.hparams.ent_coef * entropy_loss
                        + v_loss * self.hparams.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.hparams.max_grad_norm
                    )
                    self.optimizer.step()

                if self.hparams.target_kl is not None:
                    if approx_kl > self.hparams.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar(
                "charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step
            )
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar(
                "losses/explained_variance", explained_var, global_step
            )
            print("SPS:", int(global_step / (time.time() - start_time)))
            self.writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

        self.envs.close()
        self.writer.close()
