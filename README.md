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

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
#### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/bidimensional/Banana-Navigator.git
cd Banana-Navigator
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

    
### Repository Contents

- **Navigation.ipynb** - this is the all-in-one notebook to train and solve the project
- **model.pt** - final saved weights after training. This has reached an average score 14.38 after 13.000 episodes.

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

### Instructions

Follow all the steps in `Navigation.ipynb` to train and visualize the score improvements over time.

