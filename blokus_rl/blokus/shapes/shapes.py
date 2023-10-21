from abc import ABC, abstractmethod

import numpy as np


class Shape(ABC):
    """
    A class that defines the functions associated
    with a shape.
    """

    label: str = ""
    points: list = []
    corners: list = []
    idx: int = -1
    rotation_matrix = np.array([(0, -1), (1, 0)])
    ref_point: tuple[int, int] | None = None

    @abstractmethod
    def set_points(self, x, y):
        pass

    @property
    def size(self):
        return len(self.points)

    def rotate(self):
        """
        Returns the points that would be covered by a
        shape that is rotated 0, 90, 180, of 270 degrees
        in a clockwise direction.
        """
        np_ref = np.array([self.ref_point])
        np_points = np.array(self.points)
        np_corners = np.array(self.corners)

        np_points = (np_points - np_ref) @ self.rotation_matrix + np_ref
        self.points = list(map(tuple, np_points))

        np_corners = (np_corners - np_ref) @ self.rotation_matrix + np_ref
        self.corners = list(map(tuple, np_corners))

    def flip(self):
        """
        Returns the points that would be covered if the shape
        was flipped horizontally or vertically.
        """
        np_ref = np.array([self.ref_point])
        np_points = np.array(self.points)
        np_corners = np.array(self.corners)

        np_points = np_points - np_ref
        np_points[:, 1] = -np_points[:, 1]
        np_points = np_points + np_ref
        self.points = list(map(tuple, np_points))

        np_corners = np_corners - np_ref
        np_corners[:, 1] = -np_corners[:, 1]
        np_corners = np_corners + np_ref
        self.corners = list(map(tuple, np_corners))

    @staticmethod
    def from_json(obj):
        shape = Shape()
        shape.label = obj["label"]
        shape.points = list(map(tuple, obj["points"]))
        shape.corners = list(map(tuple, obj["corners"]))
        shape.idx = obj["idx"]

        return shape

    def to_json(self, idx):
        self.idx = idx
        return {
            "label": self.label,
            "points": [(int(x), int(y)) for x, y in self.points],
            "corners": [(int(x), int(y)) for x, y in self.corners],
            "idx": self.idx,
        }

    def __eq__(self, value):
        return sorted(self.points) == sorted(value.points)

    def __lt__(self, value):
        return self.idx < value.idx

    def __hash__(self):
        return hash(str(sorted(self.points)))

    def __str__(self):
        return "\n".join([f"Id: {self.label}", f"Points: {sorted(self.points)}"])

    def __repr__(self) -> str:
        return " ".join([f"Shape: {self.label}", f"Points: {sorted(self.points)}"])


class I1(Shape):
    label = "I1"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y)]
        self.corners = [(x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)]


class I2(Shape):
    label = "I2"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 2), (x - 1, y + 2)]


class I3(Shape):
    label = "I3"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 3), (x - 1, y + 3)]


class I4(Shape):
    label = "I4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x, y + 3)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 4), (x - 1, y + 4)]


class I5(Shape):
    label = "I5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x, y + 3), (x, y + 4)]
        self.corners = [(x - 1, y - 1), (x + 1, y - 1), (x + 1, y + 5), (x - 1, y + 5)]


class V3(Shape):
    label = "V3"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y)]
        self.corners = [
            (x - 1, y - 1),
            (x + 2, y - 1),
            (x + 2, y + 1),
            (x + 1, y + 2),
            (x - 1, y + 2),
        ]


class L4(Shape):
    label = "L4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x + 1, y)]
        self.corners = [
            (x - 1, y - 1),
            (x + 2, y - 1),
            (x + 2, y + 1),
            (x + 1, y + 3),
            (x - 1, y + 3),
        ]


class Z4(Shape):
    label = "Z4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x - 1, y)]
        self.corners = [
            (x - 2, y - 1),
            (x + 1, y - 1),
            (x + 2, y),
            (x + 2, y + 2),
            (x - 1, y + 2),
            (x - 2, y + 1),
        ]


class O4(Shape):
    label = "O4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)]
        self.corners = [(x - 1, y - 1), (x + 2, y - 1), (x + 2, y + 2), (x - 1, y + 2)]


class L5(Shape):
    label = "L5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x + 2, y), (x + 3, y)]
        self.corners = [
            (x - 1, y - 1),
            (x + 4, y - 1),
            (x + 4, y + 1),
            (x + 1, y + 2),
            (x - 1, y + 2),
        ]


class T5(Shape):
    label = "T5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x - 1, y), (x + 1, y)]
        self.corners = [
            (x + 2, y - 1),
            (x + 2, y + 1),
            (x + 1, y + 3),
            (x - 1, y + 3),
            (x - 2, y + 1),
            (x - 2, y - 1),
        ]


class V5(Shape):
    label = "V5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x, y + 2), (x + 1, y), (x + 2, y)]
        self.corners = [
            (x - 1, y - 1),
            (x + 3, y - 1),
            (x + 3, y + 1),
            (x + 1, y + 3),
            (x - 1, y + 3),
        ]


class N(Shape):
    label = "N"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 2, y), (x, y - 1), (x - 1, y - 1)]
        self.corners = [
            (x + 1, y - 2),
            (x + 3, y - 1),
            (x + 3, y + 1),
            (x - 1, y + 1),
            (x - 2, y),
            (x - 2, y - 2),
        ]


class Z5(Shape):
    label = "Z5"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 1, y + 1), (x - 1, y), (x - 1, y - 1)]
        self.corners = [
            (x + 2, y - 1),
            (x + 2, y + 2),
            (x, y + 2),
            (x - 2, y + 1),
            (x - 2, y - 2),
            (x, y - 2),
        ]


class T4(Shape):
    label = "T4"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x - 1, y)]
        self.corners = [
            (x + 2, y - 1),
            (x + 2, y + 1),
            (x + 1, y + 2),
            (x - 1, y + 2),
            (x - 2, y + 1),
            (x - 2, y - 1),
        ]


class P(Shape):
    label = "P"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x + 1, y), (x + 1, y - 1), (x, y - 1), (x, y - 2)]
        self.corners = [
            (x + 1, y - 3),
            (x + 2, y - 2),
            (x + 2, y + 1),
            (x - 1, y + 1),
            (x - 1, y - 3),
        ]


class W(Shape):
    label = "W"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x - 1, y), (x - 1, y - 1)]
        self.corners = [
            (x + 1, y - 1),
            (x + 2, y),
            (x + 2, y + 2),
            (x - 1, y + 2),
            (x - 2, y + 1),
            (x - 2, y - 2),
            (x, y - 2),
        ]


class U(Shape):
    label = "U"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x, y - 1), (x + 1, y - 1)]
        self.corners = [
            (x + 2, y - 2),
            (x + 2, y),
            (x + 2, y + 2),
            (x - 1, y + 2),
            (x - 1, y - 2),
        ]


class F(Shape):
    label = "F"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y + 1), (x, y - 1), (x - 1, y)]
        self.corners = [
            (x + 1, y - 2),
            (x + 2, y),
            (x + 2, y + 2),
            (x - 1, y + 2),
            (x - 2, y + 1),
            (x - 2, y - 1),
            (x - 1, y - 2),
        ]


class X(Shape):
    label = "X"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]
        self.corners = [
            (x + 1, y - 2),
            (x + 2, y - 1),
            (x + 2, y + 1),
            (x + 1, y + 2),
            (x - 1, y + 2),
            (x - 2, y + 1),
            (x - 2, y - 1),
            (x - 1, y - 2),
        ]


class Y(Shape):
    label = "Y"

    def __init__(self):
        self.set_points(0, 0)

    def set_points(self, x, y):
        self.ref_point = (x, y)
        self.points = [(x, y), (x, y + 1), (x + 1, y), (x + 2, y), (x - 1, y)]
        self.corners = [
            (x + 3, y - 1),
            (x + 3, y + 1),
            (x + 1, y + 2),
            (x - 1, y + 2),
            (x - 2, y + 1),
            (x - 2, y - 1),
        ]


def get_all_shapes() -> list[Shape]:
    return [
        I1(),
        I2(),
        I3(),
        I4(),
        I5(),
        V3(),
        L4(),
        Z4(),
        O4(),
        L5(),
        T5(),
        V5(),
        N(),
        Z5(),
        T4(),
        P(),
        W(),
        U(),
        F(),
        X(),
        Y(),
    ]
