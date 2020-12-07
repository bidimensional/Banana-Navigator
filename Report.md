# Banana-Navigator

### Introduction

This is my implementation to solve the Banana Navigation project on the Udacity Reinforcement Learning Nanodegree.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

My code has a cap of 15 for the Avg Score and 2000 episodes, but all my executions passed the threshold of 13.

This has been implemented using a Dueling Network Architecture with Epsilon-Greedy policy and an Experience Replay.

### Model Architecture
The model architecture for the Dueling DQN I implemented is quite straightforward and it is working well in practice, and you can read the research paper at the following address: https://arxiv.org/pdf/1511.06581.pdf

The architecture is very simple:
  - **1 Input Layer**
    - 37 IN / 64 OUT)
    
  - **Action advantage** and **Stage value** 
    - Layer 1 (ReLU Activated) :  64 IN / 32 OUT
    - Layer 2:                    32 IN / 1 OUT

### Hyperparameters
These are the parameters I used. They are pretty standard and really haven't played around them too much, with the exception of the Larning Rate is one I had to shrink as the agent wasn't learning initially. This smaller value helped.

### Agent
BUFFER_SIZE = int(1e5)  \
BATCH_SIZE = 64  
GAMMA = 0.99 \
TAU = 1e-3  \
LR = 0.0001 \
UPDATE_EVERY = 10  \

### Training
eps_start=1.0 \
eps_end=0.01 \
eps_decay=0.995 




