"""Monte Carlo Tree Search implementation."""
import math

import numpy as np


class MCTS:
    def __init__(self, game, nn):
        self.game = game
        self.nn = nn
        self.tree = {}

    def simulate(
        self,
        s,
        current_player: int,
        cpuct: int = 1,
        epsilon_fix: bool = True,
    ):
        """Simulate a game from the current state.

        Actions are selected according to the upper confidence bound.
        The tree is expanded as the simulation progresses.
        Heuristic scores are updated at the end of the simulation.

        Args:
            s: The current state.
            current_player: The current player.
            terminal: Whether the current state is terminal.
            winners: The winners of the game.
            cpuct: The exploration constant.
            epsilon_fix: Whether to use the epsilon fix.

        Returns:
            The scores for each player."""
        # Get hash of state
        hashed_s = self.game.string_representation(s)

        if hashed_s in self.tree:  # Not at leaf; select.
            # Select action with highest upper confidence bound.
            stats = self.tree[hashed_s]
            N, Q, P = stats[:, 1], stats[:, 2], stats[:, 3]
            U = cpuct * P * math.sqrt(N.sum() + (1e-6 if epsilon_fix else 0)) / (1 + N)
            heuristic = Q + U
            best_a_idx = np.argmax(heuristic)
            best_a = stats[best_a_idx, 0]  # Pick best action to take
            s_prime, current_player = self.game.get_next_state(
                s, current_player, best_a[0]
            )
            scores = self.simulate(
                s_prime, current_player
            )  # Forward simulate with this action
            n, q = N[best_a_idx], Q[best_a_idx]
            v = scores[current_player]  # Index in to find our reward
            stats[best_a_idx, 2] = (n * q + v) / (n + 1)
            stats[best_a_idx, 1] += 1
            return scores

        else:  # Expand
            winners = self.game.get_game_ended(s)
            if winners is not None:
                return winners
            available_actions = self.game.get_valid_moves(s, current_player)
            idx = np.stack(np.where(available_actions)).T
            obs, mask = self.game.get_observation(s, current_player)
            p, v = self.nn.predict(obs, mask)
            stats = np.zeros((len(idx), 4), dtype=np.object_)
            stats[:, -1] = p
            stats[:, 0] = list(idx)
            self.tree[hashed_s] = stats
            return v

    def get_distribution(self, s, temperature):
        """Get the MCTS policy distribution for a given state.

        Args:
            s: The state.
            temperature: The temperature.

        Returns:
            The MCTS policy distribution."""
        hashed_s = self.game.string_representation(s)
        stats = self.tree[hashed_s][:, :2].copy()
        N = stats[:, 1]
        try:
            raised = np.power(N, 1 / temperature)
        # As temperature approaches 0, the effect becomes equivalent to argmax.
        except (ZeroDivisionError, OverflowError):
            raised = np.zeros_like(N)
            raised[N.argmax()] = 1

        total = raised.sum()
        # If all children are unexplored, prior is uniform.
        if total == 0:
            raised[:] = 1
            total = raised.sum()
        dist = raised / total
        stats[:, 1] = dist
        return stats
