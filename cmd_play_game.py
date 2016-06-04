from __future__ import print_function
__author__ = 'frankhe'

import time
from Parameters import Test
import curses
stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
stdscr.keypad(1)
stdscr.nodelay(1)

"""
Press s to start console, and press q to end the console. Press numbers to take actions
"""
game_name = Test.game_name
print_states = False

import gym
env = gym.make(game_name)
# env.monitor.start('MsPacman-exp')
env.reset()
game_episodes = [{'reward':0}]

while True:
    if stdscr.getch() == ord('s'):
        break

print("Playing ", game_name, end=' ')
print("action_space=", env.action_space, end=' ')
print("observation_space=", env.observation_space, end=' ')

action_dict = [ord(str(x)) for x in range(9)]

action = 0

while True:
    env.render()
    # action = env.action_space.sample()
    c = stdscr.getch()
    if c == ord('q'):
        break
    for action_i in range(len(action_dict)):
        if action_dict[action_i] == c:
            action = action_i
    if c != curses.ERR:
        print('action=', action, end='')

    observation, reward, done, info = env.step(action)  # take a action
    time.sleep(0.1)

    episode = len(game_episodes)-1
    game_episodes[episode]['reward'] += reward

    if done:
        game_episodes.append({'reward':0})
        print('game ', episode, ' is done! Reward=', game_episodes[episode]['reward'], end='')
        env.reset()

    if print_states:
        print(_, end='')
        print("action=", action, end='')
        print("observation=", observation.shape, end='')
        print("reward=", reward, end='')
        print("done=", done, end='')
        print("info=", info, end='')

curses.endwin()
for episode in game_episodes:
    print('reward=', episode['reward'])
# env.monitor.close()
