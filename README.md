# Blokus-RL
Implementation of a reinforcement learning model for the strategy game [Blokus](https://scheherazade.znadplanszy.pl/2018/03/31/blokus/)

## Installation
To install the required packages, run the following command:
```bash
pip install -e .
```

## Usage

### Training

This repository allows you to train a reinforcement learning model to play Blokus and other games using the [OpenAI Gym](https://gym.openai.com/) interface. Currently, there are two algorithms implemented: [PPO](https://arxiv.org/abs/1707.06347) and [AlphaZero](https://arxiv.org/abs/1712.01815). Before training, you need to create a configuration file. You can find an example in `config` directory. For more information about hyperparameters, see `blokus_rl/hparams.py` there are described all hyperparameters for both algorithms. To train a model, run the following command:
```bash
python -m blokus_rl.train \
        --hparams_fp <path_to_config_file> \
        --algorithm <ppo|alphazero>
```
The training logs and checkpoints will be saved as specified in the config file directory. The default directory is `models/checkpoints` and `models/logs` respectively. To see the training progress, run the following command:
```bash
tensorboard --logdir <path_to_logs_directory>
```

### Compare models

To compare two agents, you can run the following command:
```bash
python -m blokus_rl.compare_arena <path_to_config_file> 
```
