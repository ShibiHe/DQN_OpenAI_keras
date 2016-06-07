__author__ = 'frankhe'
""" this script shows the actions in openAI gym are different from the actions in DQN paper which used ALE too"""
""" https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py (read line 16 61 139) """
from atari_py import *
ale = ALEInterface()
ale.loadROM(get_game_path('breakout'))
print(ale.getMinimalActionSet())

ale.loadROM(get_game_path('ms_pacman'))
print(ale.getMinimalActionSet())
"""
Here are the results in this script:a
[ 0  1  3  4 11 12]
[0 2 3 4 5 6 7 8 9]
!! notice that the actions are not from 0 to n-1, therefore the sample() function in
https://github.com/openai/gym/blob/master/gym/spaces/discrete.py is probably not right!!
UPDATE: after checking the source code the function is right, and OpenAI uses frame skipping originally.

Next are the results in DQN paper: (The differences are in line 59, 137)

parallels@ubuntu:~/Github/DeepMind-Atari-Deep-Q-Learner$ ./run_cpu breakout
-framework alewrap -game_path /home/parallels/Github/DeepMind-Atari-Deep-Q-Learner/roms/ -name DQN3_0_1_breakout_FULL_Y -env breakout -env_params useRGB=true -agent NeuralQLearner -agent_params lr=0.00025,ep=1,ep_end=0.1,ep_endt=replay_memory,discount=0.99,hist_len=4,learn_start=50000,replay_memory=1000000,update_freq=4,n_replay=1,network="convnet_atari3",preproc="net_downsample_2x_full_y",state_dim=7056,minibatch_size=32,rescale_r=1,ncols=1,bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1 -steps 50000000 -eval_freq 250000 -eval_steps 125000 -prog_freq 5000 -save_freq 125000 -actrep 4 -gpu -1 -random_starts 30 -pool_frms type="max",size=2 -seed 1 -threads 4
Torch Threads:	4
Using CPU code only. GPU device id:	-1
Torch Seed:	1

Playing:	breakout


Today's debug started here!


Creating Agent Network from convnet_atari3
____________________________________________
convent_atari3.lua
actions	table: 0x41508978	table
bestq	0	number
bufferSize	512	number
clip_delta	1	number
discount	0.99	number
ep	1	number
ep_end	0.1	number
ep_endt	1000000	number
ep_start	1	number
filter_size	table: 0x4150a5d0	table
filter_stride	table: 0x4150a620	table
gpu	-1	number
histSpacing	1	number
histType	linear	string
hist_len	4	number
input_dims	table: 0x415089f8	table
learn_start	50000	number
lr	0.00025	number
lr_end	0.00025	number
lr_endt	1000000	number
lr_start	0.00025	number
max_reward	1	number
min_reward	-1	number
minibatch_size	32	number
n_actions	4	number   !!!!!!!!!!!!!!!!!!!!!!!!! The difference !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
n_hid	table: 0x4150a670	table
n_replay	1	number
n_units	table: 0x4150a580	table
ncols	1	number
network	function: 0x4150a518	function
nl	table: 0x40679aa8	table
nonTermProb	1	number
preproc	net_downsample_2x_full_y	string
replay_memory	1000000	number
rescale_r	1	number
state_dim	7056	number
target_q	10000	number
transition_params	table: 0x41508a48	table
update_freq	4	number
valid_size	500	number
verbose	10	number
wc	0	number
____________________________________________

nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> output]
  (1): nn.Reshape(4x84x84)
  (2): nn.SpatialConvolution(4 -> 32, 8x8, 4,4, 1,1)
  (3): nn.Rectifier
  (4): nn.SpatialConvolution(32 -> 64, 4x4, 2,2)
  (5): nn.Rectifier
  (6): nn.SpatialConvolution(64 -> 64, 3x3)
  (7): nn.Rectifier
  (8): nn.Reshape(3136)
  (9): nn.Linear(3136 -> 512)
  (10): nn.Rectifier
  (11): nn.Linear(512 -> 4)
}
Convolutional layers flattened output size:	3136




parallels@ubuntu:~/Github/DeepMind-Atari-Deep-Q-Learner$ ./run_cpu pacman
-framework alewrap -game_path /home/parallels/Github/DeepMind-Atari-Deep-Q-Learner/roms/ -name DQN3_0_1_pacman_FULL_Y -env pacman -env_params useRGB=true -agent NeuralQLearner -agent_params lr=0.00025,ep=1,ep_end=0.1,ep_endt=replay_memory,discount=0.99,hist_len=4,learn_start=50000,replay_memory=1000000,update_freq=4,n_replay=1,network="convnet_atari3",preproc="net_downsample_2x_full_y",state_dim=7056,minibatch_size=32,rescale_r=1,ncols=1,bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1 -steps 50000000 -eval_freq 250000 -eval_steps 125000 -prog_freq 5000 -save_freq 125000 -actrep 4 -gpu -1 -random_starts 30 -pool_frms type="max",size=2 -seed 1 -threads 4
Torch Threads:	4
Using CPU code only. GPU device id:	-1
Torch Seed:	1

Playing:	pacman


Today's debug started here!


Creating Agent Network from convnet_atari3
____________________________________________
convent_atari3.lua
actions	table: 0x40401a78	table
bestq	0	number
bufferSize	512	number
clip_delta	1	number
discount	0.99	number
ep	1	number
ep_end	0.1	number
ep_endt	1000000	number
ep_start	1	number
filter_size	table: 0x404036f0	table
filter_stride	table: 0x40403740	table
gpu	-1	number
histSpacing	1	number
histType	linear	string
hist_len	4	number
input_dims	table: 0x40401b18	table
learn_start	50000	number
lr	0.00025	number
lr_end	0.00025	number
lr_endt	1000000	number
lr_start	0.00025	number
max_reward	1	number
min_reward	-1	number
minibatch_size	32	number
n_actions	5	number           !!!!!!!!!!!!!!!!!!!!!!!!! The difference !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
n_hid	table: 0x40403790	table
n_replay	1	number
n_units	table: 0x404036a0	table
ncols	1	number
network	function: 0x40403638	function
nl	table: 0x40da8b80	table
nonTermProb	1	number
preproc	net_downsample_2x_full_y	string
replay_memory	1000000	number
rescale_r	1	number
state_dim	7056	number
target_q	10000	number
transition_params	table: 0x40401b68	table
update_freq	4	number
valid_size	500	number
verbose	10	number
wc	0	number
____________________________________________

nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> output]
  (1): nn.Reshape(4x84x84)
  (2): nn.SpatialConvolution(4 -> 32, 8x8, 4,4, 1,1)
  (3): nn.Rectifier
  (4): nn.SpatialConvolution(32 -> 64, 4x4, 2,2)
  (5): nn.Rectifier
  (6): nn.SpatialConvolution(64 -> 64, 3x3)
  (7): nn.Rectifier
  (8): nn.Reshape(3136)
  (9): nn.Linear(3136 -> 512)
  (10): nn.Rectifier
  (11): nn.Linear(512 -> 5)
}
Convolutional layers flattened output size:	3136
____________________________________________

"""