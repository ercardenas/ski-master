# ski-master
CS221 Project - Reinforcement Learning (RL) in Atari 2600 Skiing game environment.

## Getting Started

```
git clone git@github.com:ercardenas/ski-master.git

conda create -n ski-master python=2.7 anaconda
conda activate ski-master

pip install gym
pip install gym[atari]

pip install pygame
pip install matplotlib

pip install cv2
pip install opencv-python
```

## How to run the code?

After activating the evinroment, there are three different models which you can run:

1. Heuristic Based Model:

```
python train_heuristic.py
```

2. Imaged Based Q-Learning:

```
python train.py
```
