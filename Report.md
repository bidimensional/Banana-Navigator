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

### My results
```
Episode 100	Average Score: 0.08
Episode 200	Average Score: 1.09
Episode 300	Average Score: 3.11
Episode 400	Average Score: 6.50
Episode 500	Average Score: 9.30
Episode 600	Average Score: 10.68
Episode 700	Average Score: 10.52
Episode 800	Average Score: 8.71
Episode 900	Average Score: 10.75
Episode 1000	Average Score: 12.56
Episode 1100	Average Score: 13.64
Episode 1200	Average Score: 14.47
Episode 1300	Average Score: 14.38

Environment Solved.
```
Reference-style: 
![graph]

[graph]: https://raw.githubusercontent.com/bidimensional/Banana-Navigator/main/Screenshot_from_2020-12-02_12.01.52.png


### Model Architecture
The model architecture for the Dueling DQN I implemented is quite straightforward and it is working well in practice. It is based on the findings of the original research paper at the following address: https://arxiv.org/pdf/1511.06581.pdf

The architecture is very simple:
  - **1 Input Layer**
    - 37 IN / 64 OUT)
    
  - **Action advantage** and **Stage value** 
    - Layer 1 (ReLU Activated) :  64 IN / 32 OUT
    - Layer 2:                    32 IN / 1 OUT


### Learning algorithm
The agent learns maximising the reward it gets (+1 for each yellow banana, -1 for each blue banana) through iterations of the episodic task until the average score of 13 and above is achieved or a maximum number of episodes has been reached. It is using  epsilon-greedy decay to always keep a degree of exploration, from eps_start down to a bottom limit as defined by the eps_end parameter (see below).

Adam is used as an optimizer using the LR specified below.

The training creates a **model.pt** when successful that can be used to restore the weights in the neural network at later stage to let the agent interact with the world.

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

### Future research
As I experienced high variance in the results, the first thing I would fine tune are the hyperparameters, in particular the Learning Rate and see how the train preforms.
The Neural Network used for the model is also very simple, and that would be the next area I would investigate, to see whether adding more neurons and/or more layers helps the agent learning more complex strategies for this specific task.


