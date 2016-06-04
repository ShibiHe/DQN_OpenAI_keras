from __future__ import print_function
__author__ = 'frankhe'

import numpy as np
import image_preprocessing as IP
import gym


env = gym.make('MsPacman-v0')
# env = gym.make('AirRaid-v0')

env.reset()
print("action_space=", env.action_space)
print("observation_space=", env.observation_space)

print_states = True

images = []

for _ in range(4):
    env.render()
    action = env.action_space.sample()
    if action == 0:
        action = 1
    observation, reward, done, info = env.step(action)  # take a random action
    images.append(observation)
    if print_states:
        print(_)
        print("action=", action, 'reward=', reward, 'done=', done)
        # print "observation=", observation.shape
        # time.sleep(0.1)
        # print "info=", info
        # raw_input()
    if done:
        env.reset()

images = map(IP.rgb2gray, images)
import cPickle
file1 = open('./images_test', mode='wb+')
cPickle.dump(images, file1, 2)
# IP.imshow(images[0])