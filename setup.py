from setuptools import setup, find_packages

setup(
    name="blokus-rl",
    version="0.0.0",
    author="Jakub Kubiak",
    author_email="kubiak.jakub01@gmail.com",
    description="Reinforcement learning for blokus game",
    packages=find_packages(),
    install_requires=[
        "colosseumrl@git+https://github.com/KubiakJakub01/colosseumrl#egg=master",
        "Cython==3.0.3",
        "PyYAML==6.0.1",
        "coloredlogs==15.0.1",
        "gymnasium==0.29.1",
        "matplotlib==3.8.0",
        "moviepy==1.0.3",
        "mypy==1.6.1",
        "numpy==1.25.2",
        "optuna==3.4.0",
        "pycodestyle==2.11.1",
        "pylint==3.0.1",
        "pytablewriter==1.2.0",
        "PyYAML==6.0.1",
        "tensorboard==2.14.1",
        "torch==2.0.1",
        "torch-summary==1.4.5",
        "types-PyYAML==6.0.12.12",
        "wandb==0.15.12"
    ],
    extras_require={
        "dev": [
            "blokus-gym@git+https://github.com/KubiakJakub01/blokus_ai#egg=master",
        ]
    },
    classifiers = [
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Reinforcement Learning",
        "Topic :: Games/Entertainment :: Board Games"
    ],
)
