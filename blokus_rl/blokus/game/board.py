"""Module for the Board class."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..players.player import Player
from ..shapes.shape import Shape

plt.switch_backend('agg')

class Board:
    """
    Creates a board that has n rows and
    m columns with an empty space represented
    by a character string according to null of
    character length one.
    """

    def __init__(self, size):
        plt.ion()
        self.size = size
        self.tensor = torch.zeros((size, size), dtype=torch.int32)

    def update(self, player: Player, move: Shape):
        """
        Takes in a Player object and a move as a
        list of integer tuples that represent the piece.
        """
        for x, y in move.points:
            self.tensor[y][x] = player.index

    def in_bounds(self, point: tuple[int, int]):
        """
        Takes in a tuple and checks if it is in the bounds of
        the board.
        """
        x, y = point
        return 0 <= x < self.size and 0 <= y < self.size

    def overlap(self, points: list[tuple[int, int]]):
        """
        Returns a boolean for whether a move is overlapping
        any pieces that have already been placed on the board.
        """
        return any(self.tensor[y][x].item() != 0 for x, y in points)

    def is_player_tile(self, player: Player, point: tuple[int, int]):
        x, y = point
        return self.in_bounds((x, y)) and self.tensor[y][x].item() == player.index

    def corner(self, player: Player, move: Shape):
        """
        Note: ONLY once a move has been checked for adjacency, this
        function returns a boolean; whether the move is cornering
        any pieces of the player proposing the move.
        """
        return any(
            self.is_player_tile(player, (x + 1, y + 1))
            or self.is_player_tile(player, (x - 1, y - 1))
            or self.is_player_tile(player, (x - 1, y + 1))
            or self.is_player_tile(player, (x + 1, y - 1))
            for x, y in move.points
        )

    def adj(self, player: Player, move: Shape):
        """
        Checks if a move is adjacent to any squares on
        the board which are occupied by the player
        proposing the move and returns a boolean.
        """
        return any(
            self.is_player_tile(player, (x, y + 1))
            or self.is_player_tile(player, (x, y - 1))
            or self.is_player_tile(player, (x - 1, y))
            or self.is_player_tile(player, (x + 1, y))
            for x, y in move.points
        )

    def print_board(self, mode="human"):
        if mode == "human":
            self.fancy_board()
        elif mode == "minimal":
            self.print_board_min()
        elif mode == "tensor":
            self.print_tensor()

    def print_tensor(self):
        print(chr(27) + "[2J")
        print(self.tensor.permute())

    def print_board_min(self):
        print(chr(27) + "[2J")
        all_non_zeros = self.tensor != 0
        coverage = (
            all_non_zeros.sum().item()
            / (all_non_zeros.shape[0] * all_non_zeros.shape[1])
            * 100
        )
        print(f"Coverage: {coverage:.2f}%")

    def fancy_board(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        colors = {0: "lightgrey", 1: "red", 2: "blue", 3: "yellow", 4: "green"}

        for y in range(self.size):
            for x in range(self.size):
                polygon = plt.Polygon(
                    [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]]
                )
                polygon.set_facecolor(colors[self.tensor[y][x].item()])
                ax.add_patch(polygon)

        plt.yticks(np.arange(0, self.size, 1))
        plt.xticks(np.arange(0, self.size, 1))
        plt.grid()

        # Render the image and convert it to a NumPy array
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)  # Close the figure to free up resources

        return image_data
