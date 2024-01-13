# Blokus-RL
Implementation of a reinforcement learning model for the strategy game [Blokus](https://scheherazade.znadplanszy.pl/2018/03/31/blokus/)

## Installation

### Cuda

This repository requires [CUDA](https://developer.nvidia.com/cuda-zone) to be installed. You can find installation instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). To check if CUDA is installed correctly, run the following command:
```bash
nvidia-smi
```

### Conda
This repository requires [Conda](https://docs.conda.io/en/latest/) to be installed. You can find installation instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). To check if Conda is installed correctly, run the following command:
```bash
conda --version
```

### Conda environment
To create a Conda environment with all required packages, run the following command:
```bash
conda env create -n blokus-rl python=3.10
```

To activate the environment, run the following command:
```bash
conda activate blokus-rl
```

### Required packages
To install the required packages, run the following command:
```bash
git clone https://github.com/KubiakJakub01/Blokus-RL.git
cd Blokus-RL
pip install -e .[dev]
```

### Docker
Alternatively, you can use [Docker](https://www.docker.com/) to run this repository. To build a Docker image, run the following command from the root directory of this repository:
```bash
docker build -t blokus-rl .
```

When running the Docker image remember to mount proper directories with config files, logs, and checkpoints. For example, to run a sample PPO training in Blokus $7$x$7$ environment, you can use the following command:
```bash
docker run -it --rm \
           -v $(pwd)/config:/app/config \
           blokus-rl \
           python -m blokus_rl.train \
                  --hparams_fp config/ppo_blokus_7x7.yml \
                  --algorithm ppo
```

## Usage

### Training

This repository allows you to train a reinforcement learning model to play Blokus and other games using the [OpenAI Gym](https://gym.openai.com/) interface. Currently, there are two algorithms implemented: [PPO](https://arxiv.org/abs/1707.06347) and [AlphaZero](https://arxiv.org/abs/1712.01815). Before training, you need to create a file with `hparams`. You can find an example in `config` directory. For more information about hyperparameters, see `blokus_rl/hparams.py` there are described all hyperparameters for both algorithms. To train a model, run the following command:
```bash
python -m blokus_rl.train \
        --hparams_fp <path_to_config_file> \
        --algorithm <ppo|alphazero>
```

#### PPO

For running sample PPO training in Blokus $7$x$7$ environment, you can use the following command:
```bash
python -m blokus_rl.train \
        --hparams_fp config/ppo_blokus_7x7.yml \
        --algorithm ppo
```

#### AlphaZero

For running sample AlphaZero training in Blokus $20$x$20$ environment, you can use the following command:
```bash
python -m blokus_rl.train \
        --hparams_fp config/alphazero_blokus_20x20.yml \
        --algorithm alphazero
```

#### Tensorboard

The training logs and checkpoints will be saved as specified in the config file directory. The default directory is `models/checkpoints` and `models/logs` respectively. To see the training progress, run the following command:
```bash
tensorboard --logdir <path_to_logs_directory>
```

### Compare models

To compare two agents, you can run the following command:
```bash
python -m blokus_rl.compare_arena <path_to_config_file> 
```
