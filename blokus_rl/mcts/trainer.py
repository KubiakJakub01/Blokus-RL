import copy
import time
from collections import defaultdict
from pickle import Pickler, Unpickler
from statistics import mean
from typing import Any, Literal

import imageio
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytablewriter import MarkdownTableWriter

from ..arena import play_match
from ..colossumrl import ColosseumBlokusGameWrapper, BlokusNNet
from ..hparams import MCTSHparams
from ..neural_network import BlokusNNetWrapper
from ..players import MCTSPlayer
from ..utils import LOG_DEBUG, LOG_INFO, LOG_WARNING
from .arena import Arena
from .dataset import MCTSDataset
from .mcts import MCTS
from ..models import DumbNet


class AlphaZeroTrainer:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, hparams: MCTSHparams):
        """Initialize monte carlo tree search trainer.

        Args:
            hparams: Hyperparameters."""
        self._build_hparams(hparams)
        self.device = (
            "cuda" if torch.cuda.is_available() and self.hparams.cuda else "cpu"
        )
        self.writer = SummaryWriter(log_dir=self.hparams.log_dir)

        # Blokus game wrapper
        self.game = ColosseumBlokusGameWrapper(self.hparams)

        # Agent and opponent neural nets with monte carlo tree search
        self.nnet = BlokusNNetWrapper(self.game, self.hparams, BlokusNNet, self.device)

        # Can be overriden in load_checkpoint()
        # self.training_data = np.zeros((0,4))
        self.training_data = []
        self.interation = 0
        if self.hparams.load_checkpoint_step is not None:
            LOG_INFO("Starting from checkpoint %d", self.hparams.load_checkpoint_step)
            self._load_checkpoint(self.hparams.load_checkpoint_step)

        self.skip_first_self_play = False  # can be overriden in load_train_examples()
        self._model_version = 0
        self._running_vals = self._reset_running_vals()

        LOG_INFO("Trainer initialized with device %s", self.device)

    def train(self):
        """Train the model."""
        for i in range(self.hparams.num_iters):
            LOG_INFO("Starting iteration %d", i + 1)
            self.interation += 1
            self._run_iteration()

    # Does one game of self play and generates training samples.
    def _self_play(self, temperature):
        s, current_player = self.game.get_init_board()
        tree = MCTS(self.game, self.nnet)

        data = []
        scores = self.game.get_game_ended(s)
        root = True
        alpha = 1
        weight = 0.25
        while scores is None:
            # Think
            for _ in range(self.hparams.num_mcts_sims):
                tree.simulate(s, current_player, cpuct=self.hparams.cpuct)

            # Fetch action distribution and append training example template.
            dist = tree.get_distribution(s, temperature=temperature)

            # Add dirichlet noise to root
            if root:
                noise = np.random.dirichlet(
                    np.array(alpha * np.ones_like(dist[:, 1].astype(np.float32)))
                )
                dist[:, 1] = dist[:, 1] * (1 - weight) + noise * weight
                root = False

            obs, mask = self.game.get_observation(s, current_player)
            data.append(
                [obs, mask, dist[:, 1].astype(np.float32), None]
            )  # state, prob, outcome

            # Sample an action
            idx = np.random.choice(len(dist), p=dist[:, 1].astype(np.float32))
            a = dist[idx, 0][0]

            # Apply action
            s, current_player, terminal, winners = self.game.get_next_state(
                s, current_player, a
            )

            # Get scores
            if terminal:
                scores = self.game.get_scores(winners)

        # Update training examples with outcome
        for i, _ in enumerate(data):
            data[i][-1] = scores

        return data

    # Performs one iteration of policy improvement.
    # Creates some number of games, then updates network parameters some number of times from that training data.
    def _run_iteration(self):
        """Run one iteration of policy improvement.

        Iteration consists of self-play, training, and arena compare."""

        # Gather training examples from self-play
        for _ in tqdm(range(self.hparams.num_eps), desc="Self play"):
            new_train_data = self._self_play(self.hparams.temperature)

        # Save the training examples
        self._save_train_examples(self.interation, new_train_data)

        # Save temp model to load into pnet
        self.nnet.save_checkpoint(filename=self.hparams.temp_model_name)

        # Prepare the training data
        losses = []
        train_dl = DataLoader(
            MCTSDataset(self.hparams), batch_size=self.hparams.batch_size, shuffle=True
        )

        # Train the model
        with tqdm(total=self.hparams.num_updates, desc="Training Net") as train_bar:
            for i, batch in enumerate(self._make_infinite_dataloader(train_dl)):
                loss = self.nnet.train_step(batch)
                losses.append(loss)
                train_bar.set_postfix(loss=loss)
                train_bar.update()
                if i >= self.hparams.num_updates:
                    break

        # Log the training loss
        self._update_running_vals({"loss": losses}, prefix="train")
        LOG_INFO("Average train loss: %.2f", mean(losses))

        # Save the model
        self.nnet.save_checkpoint(
            filename=self._get_checkpoint_file(self.interation),
        )

        # Compare the models
        # LOG_INFO("Arena comparing")
        self._arena_compare(self.hparams.opponent_type)

    def _arena_compare(self, opponent_type: Literal["pnet", "random", "uninformed"]):
        """Arena compare the models."""
        # Prepare opponents
        if opponent_type == "pnet":
            opponent = BlokusNNetWrapper(
                self.game, self.hparams, BlokusNNet, self.device
            )
            opponent.load_checkpoint(filename=self.hparams.temp_model_name)
        elif opponent_type == "uninformed":
            opponent = BlokusNNetWrapper(self.game, self.hparams, DumbNet, self.device)

        agent = MCTSPlayer(
            game=self.game,
            nn=self.nnet,
            simulations=self.hparams.num_mcts_sims,
        )
        opponents = [
            MCTSPlayer(
                game=self.game,
                nn=opponent,
                simulations=self.hparams.opponent_strength,
            )
            for _ in range(3)
        ]

        # Create the arena
        scores, frames = play_match(
            self.game,
            [agent] + opponents,
            permute=False,
            capture_video=True
        )

        # Save the video
        video_fp = self.hparams.video_dir / f"colloseumrl_{self.interation}.mp4"
        LOG_INFO("Saving video to %s", str(video_fp))
        print(f"Frames: {len(frames)}")
        imageio.mimsave(video_fp, frames, fps=1)

        # Log the results
        players = [f"agent_{self.interation}"] + [f"{opponent_type}_{self.hparams.opponent_strength}"] * 3
        writer = MarkdownTableWriter(
            table_name=f"Arena compare {self.interation}",
            headers=["Player", "Score"],
            value_matrix=[(p, s) for p, s in zip(players, scores)],
        )

        writer.write_table()

        # Write to tensorboard
        self.writer.add_text(
            f"Arena compare {self.interation}", writer.dumps(), self.interation
        )

        # Log the results
        LOG_INFO("Arena compare %s: %s", opponent_type, str(scores))

    def _make_infinite_dataloader(self, train_dl: DataLoader):
        """Make infinite dataloader."""
        while True:
            for batch in train_dl:
                yield batch

    def _get_checkpoint_file(self, iteration: int):
        """Get the checkpoint file name."""
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def _get_data_file(self, iteration: int):
        """Get the data file name."""
        return "checkpoint_" + str(iteration) + ".examples"

    def _save_train_examples(self, iteration: int, train_examples):
        """Save the train examples to a file."""
        LOG_INFO("Saving train_examples to file after %d iteration", iteration)
        filename = self.hparams.data_dir / self._get_data_file(iteration)
        with open(filename, "wb+") as f:
            Pickler(f).dump(train_examples)

    def _load_checkpoint(self, iteration: int):
        """Load the checkpoint."""
        if self.hparams.load_checkpoint_step is None:
            LOG_WARNING("load_checkpoint_step is None")
            return
        model_filename = self._get_checkpoint_file(iteration)
        examples_fp = (self.hparams.checkpoint_dir / model_filename).with_suffix(
            ".examples"
        )
        if (self.hparams.checkpoint_dir / self.hparams.best_model_name).exists():
            model_filename = self.hparams.best_model_name
        assert (
            examples_fp.is_file()
        ), f"File with train_examples not found: {examples_fp}"
        self.nnet.load_checkpoint(filename=model_filename)
        self.pnet.load_checkpoint(filename=model_filename)
        LOG_INFO("Loading train_examples from file %s", str(examples_fp))
        with open(examples_fp, "rb") as f:
            self.train_examples_history = Unpickler(f).load()
        LOG_INFO("Loaded %d train_examples", len(self.train_examples_history))

        # examples based on the model were already collected (loaded)
        self.skip_first_self_play = True
        self.interation = self.hparams.load_checkpoint_step

    def _log_to_tensorboard(self, step: int):
        """Log the running values to tensorboard."""
        for key, value in self._running_vals.items():
            self.writer.add_scalar(key, mean(value), step)
        self._running_vals = self._reset_running_vals()

    def _reset_running_vals(self):
        """Reset the running values."""
        return defaultdict(list)

    def _update_running_vals(self, items: dict[str, Any], prefix=None):
        """Update the running values.

        Args:
            items: Dictionary containing the values.
            prefix: Prefix for the keys."""
        for key, value in items.items():
            if isinstance(value, Tensor):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.item()
            if prefix is None:
                self._running_vals[key].append(value)
            else:
                self._running_vals[f"{prefix}/{key}"].append(value)

    def _log_video(self, items: list[dict[str, Any]], step: int):
        """Log the video."""
        if not self.hparams.capture_video:
            return
        LOG_INFO("Logging video")
        video_dir = self.hparams.video_dir / f"step_{step}"
        video_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            frames = item["frames"]
            game_result = item["result"]
            player = item["player"]
            # Add frames as images to tensorboard
            for i, frame in enumerate(frames):
                frame = np.transpose(frame, (2, 0, 1))
                self.writer.add_image(
                    f"arena_{step}/player_{player}_result_{game_result}",
                    frame,
                    i,
                    dataformats="CHW",
                )
            # Save the video
            video_fp = video_dir / f"player_{player}.mp4"
            imageio.mimsave(video_fp, frames, fps=1)

    def _build_hparams(self, hparams: MCTSHparams):
        """Build hyperparameters."""
        self.hparams = hparams
        # Create the checkpoint and log dir if they don't exist
        self.hparams.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.hparams.log_dir.mkdir(parents=True, exist_ok=True)
        self.hparams.data_dir.mkdir(parents=True, exist_ok=True)
        if self.hparams.capture_video:
            self.hparams.video_dir.mkdir(parents=True, exist_ok=True)
        # Save the hyperparameters
        hparams.dump_to_yaml(hparams.checkpoint_dir.parent / "hparams.yaml")

    def _compute_new_elo(
        self,
        agent_elo: float,
        opponent_elo: float,
        agent_wins: int,
        opponent_wins: int,
        draws: int,
    ):
        """Compute the new elo."""
        expected_score = 1 / (1 + 10 ** ((opponent_elo - agent_elo) / 400))
        change_in_rank_from_wins = (
            self.hparams.elo_convert_rate * (1 - expected_score) * agent_wins
        )
        change_in_rank_from_losses = (
            self.hparams.elo_convert_rate * (0 - expected_score) * opponent_wins
        )
        change_in_rank_from_draws = (
            self.hparams.elo_convert_rate * (0.5 - expected_score) * draws
        )
        return (
            agent_elo
            + change_in_rank_from_wins
            + change_in_rank_from_losses
            + change_in_rank_from_draws
        )
