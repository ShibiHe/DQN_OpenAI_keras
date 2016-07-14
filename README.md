# DQN_OpenAI_keras
This is the DQN implementation written by myself using OpenAI gym and keras.

##### The project has been officially abandoned. But! I have built another more powerful project containing DQN and episodic control, please go there and have a look [Model-Free-Episodic-Control](https://github.com/ShibiHe/Model-Free-Episodic-Control)

##Description

`training.py` is the main script

`agents.py` stores DQN agent

`Parameters.py` stores training parameters

`image_preprocessing` contains image preprocessing functions

`Anti_flickering.py` shows that only amalgamate two frames odd and even is not enough to obliterate flickering.

`action_difference.py` is a script to show the difference in actions of OpenAI gym and orignial DQN paper, irrelevant to the main functions.

`indexing_test.py` is a script to describe indexing problem in network.

##Exploration and Discoveries
###Building dqn network
#### 1. fixed Q targets
I build two models for Q and Q hat. I set Q hat to be untrainable. In addition I add a disconnected_grad to Q hat like <https://github.com/sherjilozair/dqn> has done, however I think that is unnecessary.
#### 2. indexing
This problem is interesting. Given Q_S a matrix of batch_size * num_action, and A a matrix of batch_size * 1, we want to have Q_S[i, A[i]].

In numpy we can do:

	batch_size = Q_S.shape[0]
	Q_S[range(batch_size), A.reshape(batch_size)]

But in theano, compiling range(Q_S.shape[0]) will raise an error.

Two ways of solution:

1. use theano.scan
2. use a mask to do indexing like I did

Details are in `indexing_test.py`

###Anti_flickering
As `Anti_flickering.py` described, I do not think adding two frames is enough to solve the flickering problem.

###Frame Skipping
After chatting with [jietang][jietang] and [Greg Brockman][gdb] I found out that OpenAI gym has already implemented frame skipping in _step() function in
<https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py>

Another finding is action difference described in `action_difference.py`

[jietang]: https://github.com/jietang
[gdb]: https://github.com/gdb

##Notes
Sometimes we need to know current lives in atari games. So I sent a pull request to OpenAI gym. <https://github.com/openai/gym/pull/163>

##Incomplete Functions:
experience relay

Prioritized Experience Replay

double DQN

dueling DQN

##References

[Playing Atari with Deep Reinforcement Learning][link1], V. Mnih et al., NIPS Workshop, 2013.

[Human-level control through deep reinforcement learning][link2], V. Mnih et al., Nature, 2015.

[link1]: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
[link2]: http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf

##Thanks

Some repositories really gave me much inspiration. They are in
<https://github.com/stars/ShibiHe/>

