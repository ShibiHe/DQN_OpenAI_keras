from __future__ import print_function
__author__ = 'frankhe'

from Parameters import Test
parameters = Test()
from agents import *
import gym

game_env = gym.make(parameters.game_name)

parameters.game_action_space = game_env.action_space
parameters.game_observation_space = game_env.observation_space

parameters.init_network_parameters()
agent = DqnAgent(parameters.NN_parameters)

training_step = 0
for episode in range(parameters.training_episodes):
    observation = game_env.reset()
    total_reward = 0
    done = False
    frame = 0
    while not done:
        frame += 1
        training_step += 1
        if parameters.game_display:
            game_env.render()
        """ take actions at the start of the game"""
        if parameters.no_op_start and frame < parameters.no_op_start:
            action = 0
        elif parameters.random_start and frame < parameters.random_start:
            action = game_env.action_space.sample()
        else:
            """ take action from an agent """
            action = agent.act(observation)

        observation, reward, done, info = game_env.step(action)

        total_reward += reward




