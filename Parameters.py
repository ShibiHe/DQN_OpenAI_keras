__author__ = 'frankhe'


class Test(object):
    training_episodes = 1

    game_display = True

    game_list = ['Breakout-v0', 'MsPacman-v0', 'AirRaid-v0']
    game_name = game_list[1]

    game_action_space = None
    game_observation_space = None

    no_op_start = 10
    random_start = None

    remove_flickering = 4
    frame_skip = 4

    image_size = (84, 84)

    NN_parameters = None

    """ NN structure """
    def init_network_parameters(self):
        self.NN_parameters = dict(
            input_shape=(self.frame_skip,) + self.image_size,
            output_shape=self.game_action_space.n
        )
