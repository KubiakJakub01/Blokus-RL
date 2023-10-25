from setuptools import setup, Extension
from Cython.Build import build_ext, cythonize


ext_modules = [
    Extension("blokus_rl.blokus.blokus_wrapper", ["blokus_rl/blokus/blokus_wrapper.py"]),
    Extension("blokus_rl.blokus.shapes.shapes", ["blokus_rl/blokus/shapes/shapes.py"]),
    Extension("blokus_rl.blokus.players.player", ["blokus_rl/blokus/players/player.py"]),
    Extension("blokus_rl.blokus.game.blokus_game", ["blokus_rl/blokus/game/blokus_game.py"]),
    Extension("blokus_rl.blokus.game.board", ["blokus_rl/blokus/game/board.py"]),
]

for e in ext_modules:
    e.cython_directives = {"language_level": "3"}

setup(
    name="Blokus-RL",
    version="0.0.0",
    description="Blokus game with reinforcement learning agents",
    author="Jakub Kubiak",
    ext_modules=cythonize(ext_modules),
    build_ext=build_ext
)
