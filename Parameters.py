__author__ = 'frankhe'


class Test(object):
    training_episodes = 1
    game_display = True
    game_list = ['Breakout-v0', 'MsPacman-v0', 'AirRaid-v0']
    game_name = game_list[1]

    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 1000000

    game_action_space = None
    game_observation_space = None

    no_op_start = None
    random_start = 30

    remove_flickering = 4
    frame_skip = 4

    image_size = (84, 84)

    NN_parameters = None
    learning_rate = 0.00025
    discount = 0.99
    """ NN structure """
    def init_network_parameters(self):
        self.NN_parameters = dict(
            num_actions=self.game_action_space.n,
            input_shape=(self.frame_skip,) + self.image_size,
            output_shape=self.game_action_space.n,
            learning_rate=self.learning_rate,
            discount= self.discount
        )
