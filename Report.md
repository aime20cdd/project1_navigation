# Report

## Learning Algorithm

A standard [Deep Q-learning](https://en.wikipedia.org/wiki/Q-learning#Deep_Q-learning) algorithm is used to solve the enviroment. The neural network behind the algorithm uses four fully-connected layers. The layer sizes are as follows:

 1. 37 -> 128
 2. 128 -> 128
 3. 128 -> 64
 4. 64 -> 4

The first three layers are run through a Relu activation.

The hyper parameters for the DQN method are found here:

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 8        # how often to update the network
```

These were largely left unchanged from the previous example the code was taken from. Only the `UPDATE_EVERY` parameter was increased since more "steps" could occur before any reward was given due to the nature of the problem.

## Plot of Rewards

![Performance Chart](performance.png)

After hitting 522 episodes, the next 100 mainted an average score of 13.0. It looks like there's potential for higher average scores if it were to train further.

## Ideas for Future Work

In order to train the agent quicker to solve the enviroment, further experimentation with different neural network architectures could lead to advantages over this architecture. Of the "Rainbow" series of DQN improvements, I think adding prioritized experience replay could help train the agent better, especially towards the end of training. This is because once the network learns the basic "avoid blue"/"turn to yellow" experiences, it could further optimize it's behavior by balancing the cases where avoiding blue and turning to yellow might be in conflict. These cases won't happen often, but will help it achieve a higher performance, hence a prioritized experience replay.
