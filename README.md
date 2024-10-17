# Reinforcement Learning for the Fugitive Board Game
This repository implements the Fugitive board game in PettingZoo [Agent Environment Cycle (AEC)](https://pettingzoo.farama.org/api/aec/) API for reinforcement learning purpose. 

# Fugitive
Fugitive is a 2-player zero-sum game with imperfect information.
- Please see RULES.md or [Official Rules](https://docs.google.com/document/d/1GzGd9ekb38rxpj47YhAk0JHlNzXeI3LYYUq3ThgzPnU/edit?tab=t.0) for the Rules of the game
- SHIFT system/ Event system is not implemented.

# Installation
```bash
conda create -n fugitive --file environment.yml
conda activate fugitive
```

# See game play from random agents
```bash
python envs/game.py
```

# Train Self-Play PPO
```bash
python train.py
```