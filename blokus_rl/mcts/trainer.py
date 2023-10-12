import copy
from collections import defaultdict, deque
from pickle import Pickler, Unpickler
from random import shuffle
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..blokus import BlokusGameWrapper, BlokusNNet, BlokusNNetWrapper
from ..hparams import MCTSHparams
from ..utils import LOG_INFO, LOG_WARNING, AverageMeter
from .arena import Arena
from .mcts import MCTS


class MCTSTrainer:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, hparams: MCTSHparams):
        self.hparams = hparams
        self.device = (
            "cuda" if torch.cuda.is_available() and self.hparams.cuda else "cpu"
        )
        self.writer = SummaryWriter(log_dir=self.hparams.log_dir)
        # Blokus game object
        self.game = BlokusGameWrapper(
            self.hparams.board_size,
            self.hparams.number_of_players,
            self.hparams.states_dir
        )
        # agent and opponent neural nets
        self.nnet = BlokusNNetWrapper(self.game, self.hparams,  self.device)
        self.pnet = BlokusNNetWrapper(self.game, self.hparams,  self.device)
        self.mcts = MCTS(self.game, self.nnet, self.hparams)
        # history of examples from num_iters_for_train_examples_history latest iterations
        self.train_examples_history = []
        # can be overriden in load_train_examples()
        self.skip_first_self_play = False
        self._running_vals = defaultdict(AverageMeter)

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board, curPlayer = self.game.get_init_board()
        self.curPlayer = curPlayer
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.get_canonical_form(board, self.curPlayer)
            temp = int(episodeStep < self.hparams.tempThreshold)

            pi = self.mcts.get_action_prob(copy.deepcopy(canonicalBoard), temp=temp)
            sym = self.game.get_symmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.get_next_state(
                board, self.curPlayer, action
            )

            r = self.game.get_game_ended(board, self.curPlayer)

            if r != 0:
                self._update_running_vals({
                    "reward": r
                }, prefix=f"player_{self.curPlayer}")
                return [
                    (x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer)))
                    for x in trainExamples
                ]

    def train(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.hparams.numIters + 1):
            # bookkeeping
            LOG_INFO("Starting Iter #%d ...", i)
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque([], maxlen=self.hparams.maxlenOfQueue)

                for _ in tqdm(range(self.hparams.numEps), desc="Self Play"):
                    self.mcts = MCTS(
                        self.game, self.nnet, self.hparams
                    )  # reset search tree
                    iteration_train_examples += self.execute_episode()

                # save the iteration examples to the history
                self.train_examples_history.append(iteration_train_examples)

            if (
                len(self.train_examples_history)
                > self.hparams.numItersForTrainExamplesHistory
            ):
                LOG_WARNING(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.train_examples_history)}"
                )
                self.train_examples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.save_train_examples(i - 1)

            # shuffle examples before training
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(filename="temp.pth.tar")
            self.pnet.load_checkpoint(filename="temp.pth.tar")
            pmcts = MCTS(self.game, self.pnet, self.hparams)

            pi_losses, v_losses = self.train(train_examples)
            self._update_running_vals({
                "pi_loss": pi_losses,
                "v_loss": v_losses
            }, prefix="loss")
            nmcts = MCTS(self.game, self.nnet, self.hparams)

            LOG_INFO("PITTING AGAINST PREVIOUS VERSION")
            arena = Arena(
                lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.play_games(self.hparams.arenaCompare)

            self._update_running_vals({
                "pwins": pwins,
                "nwins": nwins,
                "draws": draws
            }, prefix="arena")
            self._log_to_tensorboard(i)

            LOG_INFO("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.hparams.updateThreshold
            ):
                LOG_INFO("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(
                    folder=self.hparams.checkpoint_dir, filename="temp.pth.tar"
                )
            else:
                LOG_INFO("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(
                    folder=self.hparams.checkpoint_dir, filename=self.get_checkpoint_file(i)
                )
                self.nnet.save_checkpoint(
                    folder=self.hparams.checkpoint_dir, filename="best.pth.tar"
                )

    def get_checkpoint_file(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def save_train_examples(self, iteration):
        LOG_INFO(f"Saving trainExamples to file after %d iteration", iteration)
        folder = self.hparams.checkpoint_dir
        folder.mkdir(parents=True, exist_ok=True)
        filename = (folder / self.get_checkpoint_file(iteration)).with_suffix(".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)

    def load_train_examples(self):
        modelFile = self.hparams.load_folder_file[0] / self.hparams.load_folder_file[1]
        examplesFile = modelFile.with_suffix(".examples")
        assert (
            examplesFile.is_file()
        ), f"File with trainExamples not found: {examplesFile}"
        LOG_INFO("File with trainExamples found. Loading it...")
        with open(examplesFile, "rb") as f:
            self.train_examples_history = Unpickler(f).load()
        LOG_INFO("Loading done!")

        # examples based on the model were already collected (loaded)
        self.skip_first_self_play = True

    def _log_to_tensorboard(self, step: int):
        for key, value in self._running_vals.items():
            self.writer.add_scalar(key, value.avg, step)

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
