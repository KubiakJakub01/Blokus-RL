from collections import defaultdict
from pathlib import Path
from pickle import Pickler
from statistics import mean
from typing import Any, Literal

import imageio
import numpy as np
import torch
from pytablewriter import MarkdownTableWriter
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from ..colossumrl import ColosseumBlokusGameWrapper
from ..hparams import AlphaZeroHparams
from ..neural_network import BlokusNNetWrapper
from ..players import MCTSPlayer
from ..utils import LOG_INFO, LOG_WARNING, calculate_n_parameters
from .arena import play_match
from .dataset import MCTSDataset, collate_dataset_fn
from .mcts import MCTS


class AlphaZeroTrainer:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, hparams: AlphaZeroHparams):
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
        self.nnet = BlokusNNetWrapper(self.game, self.hparams, self.device)
        self.pnet: BlokusNNetWrapper

        # Can be overriden in load_checkpoint()
        self.training_data: list = []
        self.iteration = 0
        self.train_epoch = 0
        if self.hparams.load_checkpoint_step is not None:
            LOG_INFO("Starting from checkpoint %d", self.hparams.load_checkpoint_step)
            self._load_checkpoint(self.hparams.load_checkpoint_step)

        self.skip_first_self_play = False  # can be overriden in load_train_examples()
        self._model_version = 0
        self._running_vals = self._reset_running_vals()
        self.model_num_params = calculate_n_parameters(self.nnet.model)

        LOG_INFO("AlphaZero trainer initialized with device %s", self.device)
        LOG_INFO("Model type: %s", self.nnet.model_type)
        LOG_INFO("Model parameters: %.2fM", self.model_num_params / 1e6)
        self._log_model_summary_to_tensorboard()

        if self.hparams.load_checkpoint_step is None:
            # Write the model graph to tensorboard
            dummy_input = (
                torch.zeros(self.game.get_observation_size())
                .unsqueeze(0)
                .to(self.device)
            )
            self.writer.add_graph(self.nnet.model, dummy_input)
            self.writer.add_text("model/type", self.nnet.model_type)
            self.writer.add_text(
                "model/num_params [M]", str(self.model_num_params / 1e6)
            )

    def train(self):
        """Train the model."""
        for i in range(self.hparams.num_iters):
            LOG_INFO("Starting iteration %d", i + 1)
            self.iteration += 1
            self._run_iteration()

    def _self_play(self, temperature):
        """Run one episode of self-play."""
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
            s, current_player = self.game.get_next_state(s, current_player, a)

            # Get scores
            scores = self.game.get_game_ended(s)

        # Update training examples with outcome
        for i, _ in enumerate(data):
            data[i][-1] = scores

        return data

    def _run_iteration(self):
        """Run one iteration of policy improvement.

        Iteration consists of self-play, training, and arena compare."""

        if self.hparams.skip_first_self_play:
            LOG_INFO("Skipping first self play")
            self.hparams.skip_first_self_play = False
            # Save temp model to load into pnet
            self.nnet.save_checkpoint(filename=self.hparams.temp_model_name)
        else:
            # Gather training examples from self-play
            save_dir = self.hparams.data_dir / f"iteration_{self.iteration}"
            for episode in tqdm(range(self.hparams.num_eps), desc="Self play"):
                new_train_data = self._self_play(self.hparams.temperature)
                self._save_train_examples(save_dir, episode, new_train_data)

            # Save temp model to load into pnet
            self.nnet.save_checkpoint(filename=self.hparams.temp_model_name)

        # Prepare the training data
        losses = []
        train_dl = DataLoader(
            MCTSDataset(self.hparams),
            batch_size=self.hparams.batch_size,
            collate_fn=collate_dataset_fn,
            shuffle=True,
        )
        self._update_running_vals({"num_train_examples": len(train_dl)}, prefix="train")

        # Train the model
        train_bar = tqdm(
            range(self.hparams.epochs),
            desc="Training",
            total=self.hparams.epochs,
        )
        for _ in range(self.hparams.epochs):
            self.train_epoch += 1
            epoch_loss = self._train_epoch(train_dl)
            losses.append(epoch_loss)
            train_bar.update()
            train_bar.set_postfix({"loss": epoch_loss})
            self.writer.add_scalar("train/loss", epoch_loss, self.train_epoch)
            self.writer.add_scalar(
                "lr", self.nnet.optimizer.param_groups[0]["lr"], self.train_epoch
            )
        train_bar.close()

        # Log the training loss
        LOG_INFO("Average train loss: %.2f", mean(losses))

        # Load the pnet
        self.pnet = BlokusNNetWrapper(self.game, self.hparams, self.device)
        self.pnet.load_checkpoint(filename=self.hparams.temp_model_name)

        # Compare the models
        LOG_INFO("Arena comparing")
        scores, score_table, arena_items = self._arena_compare(
            self.hparams.opponent_type
        )

        # Update the elo
        old_elo = self.nnet.elo
        self.nnet.elo = self._compute_new_elo(self.nnet.elo, self.pnet.elo, scores[0])

        # Log the elo
        LOG_INFO("Agent elo: %.2f -> %.2f", old_elo, self.nnet.elo)
        self._update_running_vals({"elo": self.nnet.elo}, prefix="train")

        # Log scores to tensorboard
        self.writer.add_text("Arena compare", score_table, self.iteration)

        # Log the video
        self._log_video(arena_items, self.iteration)

        # Log the running values
        self._log_to_tensorboard(self.iteration)

        # Save the model checkpoint
        self.nnet.save_checkpoint(
            filename=self._get_checkpoint_file(self.iteration),
        )

    def _arena_compare(self, opponent_type: Literal["pnet", "random", "uninformed"]):
        """Arena compare the models."""
        # Prepare opponents
        agent = MCTSPlayer(
            game=self.game,
            nn=self.nnet,
            simulations=self.hparams.num_mcts_sims,
        )
        opponents = [
            MCTSPlayer(
                game=self.game,
                nn=self.pnet,
                simulations=self.hparams.num_mcts_sims,
            )
            for _ in range(3)
        ]

        # Create the arena
        scores, _ = play_match(
            self.game,
            [agent] + opponents,
            games_num=self.hparams.compare_arena_games,
            permute=self.hparams.permute,
            capture_video=False,
        )
        items = None
        if self.hparams.capture_video:
            _, items = play_match(
                self.game,
                [agent] + opponents,
                games_num=1,
                permute=False,
                capture_video=True,
            )

        # Log the results
        LOG_INFO("Arena compare %s: %s", opponent_type, str(scores))

        # Prepare the table
        players = [f"agent_{self.iteration}"] + [
            f"{opponent_type}_{self.hparams.num_mcts_sims}"
        ] * 3
        writer = MarkdownTableWriter(
            table_name=f"Arena compare {self.iteration}",
            headers=["Player", "Score"],
            value_matrix=list(zip(players, scores)),
        )

        # Return the table and the video
        return scores, writer.dumps(), items

    def _train_epoch(self, train_dl: DataLoader):
        """Train for one epoch."""
        losses = []
        for batch in train_dl:
            loss = self.nnet.train_step(batch)
            losses.append(loss)
        return mean(losses)

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

    def _save_train_examples(self, save_dir: Path, iteration: int, train_examples):
        """Save the train examples to a file."""
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / self._get_data_file(iteration)
        with open(filename, "wb+") as f:
            Pickler(f).dump(train_examples)

    def _load_checkpoint(self, iteration: int):
        """Load the checkpoint."""
        if self.hparams.load_checkpoint_step is None:
            LOG_WARNING("load_checkpoint_step is None")
            return
        model_filename = self._get_checkpoint_file(iteration)
        if (self.hparams.checkpoint_dir / self.hparams.best_model_name).exists():
            model_filename = self.hparams.best_model_name
        self.nnet.load_checkpoint(filename=model_filename)

        # examples based on the model were already collected (loaded)
        self.iteration = self.hparams.load_checkpoint_step
        self.train_epoch = self.iteration * self.hparams.epochs

    def _log_to_tensorboard(self, step: int):
        """Log the running values to tensorboard."""
        for key, value in self._running_vals.items():
            self.writer.add_scalar(key, mean(value), step)
        self._running_vals = self._reset_running_vals()

    def _log_model_summary_to_tensorboard(self):
        """Log the model summary to tensorboard."""
        summary(self.nnet.model, self.nnet.model.input_dim)

    def _reset_running_vals(self):
        """Reset the running values."""
        return defaultdict(list)

    def _update_running_vals(self, items: dict[str, Any], prefix=None):
        """Update the running values.

        Args:
            items: Dictionary containing the values.
            prefix: Prefix for the keys."""
        for key, value in items.items():
            if isinstance(value, list):
                value = mean(value)
            elif isinstance(value, Tensor):
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

        # Add frames as images to tensorboard
        for num_game, item in enumerate(items, 1):
            # Unpack the items
            frames = item["frames"]
            scores = item["scores"]
            for i, frame in enumerate(frames):
                frame = np.transpose(frame, (2, 0, 1))
                self.writer.add_image(
                    f"arena_{step}/game_{num_game}_scores_{scores}",
                    frame,
                    i,
                    dataformats="CHW",
                )

            # Save the video
            video_fp = video_dir / f"arena_{step}_game_{num_game}_scores_{scores}.mp4"
            imageio.mimsave(video_fp, frames, fps=1)

    def _build_hparams(self, hparams: AlphaZeroHparams):
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
        agent_score: int,
    ):
        """Compute the new elo."""
        expected_score = 1 / (1 + 10 ** ((opponent_elo - agent_elo) / 400))
        change_in_rank_from_wins = (
            self.hparams.elo_convert_rate * (1 - expected_score) * agent_score
        )
        return agent_elo + change_in_rank_from_wins
