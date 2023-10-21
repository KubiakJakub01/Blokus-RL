"""Monte Carlo Tree Search implementation."""
import copy
import math

import numpy as np

from ..blokus import BlokusGameWrapper, BlokusNNet
from ..hparams import MCTSHparams
from ..utils import LOG_ERROR

EPS = 1e-8


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: BlokusGameWrapper, nnet: BlokusNNet, hparams: MCTSHparams):
        self.game = game
        self.nnet = nnet
        self.hparams = hparams
        self._init_store()

    def _init_store(self):
        self.Q_sa = {}  # stores Q values for s,a (as defined in the paper)
        self.N_sa = {}  # stores #times edge s,a was visited
        self.N_s = {}  # stores #times board s was visited
        self.P_s = {}  # stores initial policy (returned by neural net)

        self.E_s = {}  # stores game.getGameEnded ended for board s
        self.V_s = {}  # stores game.getValidMoves for board s

    def get_action_prob(self, canonical_board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.hparams.num_mcts_sims):
            self.search(copy.deepcopy(canonical_board))
        s = self.game.string_representation(canonical_board)
        counts = [
            self.N_sa[(s, a)] if (s, a) in self.N_sa else 0
            for a in range(self.game.get_action_size())
        ]
        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.string_representation(board)

        if s not in self.E_s:
            self.E_s[s] = self.game.get_game_ended(board, 1)
        if self.E_s[s] != 0:
            # terminal node
            return -self.E_s[s]

        # print(f's not in self.Ps: {s not in self.Ps}')
        if s not in self.P_s:
            # leaf node
            self.P_s[s], v = self.nnet.predict(board.canonical_board.copy())
            valids = self.game.get_valid_moves(board)
            self.P_s[s] = self.P_s[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.P_s[s])
            if sum_Ps_s > 0:
                self.P_s[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                LOG_ERROR("All valid moves were masked, doing a workaround.")
                self.P_s[s] = self.P_s[s] + valids
                self.P_s[s] /= np.sum(self.P_s[s])

            self.V_s[s] = valids
            self.N_s[s] = 0
            return -v

        valids = self.V_s[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Q_sa:
                    u = self.Q_sa[(s, a)] + self.hparams.cpuct * self.P_s[s][
                        a
                    ] * math.sqrt(self.N_s[s]) / (1 + self.N_sa[(s, a)])
                else:
                    u = (
                        self.hparams.cpuct
                        * self.P_s[s][a]
                        * math.sqrt(self.N_s[s] + EPS)
                    )  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Q_sa:
            self.Q_sa[(s, a)] = (self.N_sa[(s, a)] * self.Q_sa[(s, a)] + v) / (
                self.N_sa[(s, a)] + 1
            )
            self.N_sa[(s, a)] += 1

        else:
            self.Q_sa[(s, a)] = v
            self.N_sa[(s, a)] = 1

        self.N_s[s] += 1
        return -v
