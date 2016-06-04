__author__ = 'frankhe'
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, Flatten, Dense


class DQN_Agent(object):
    def __init__(self, parameters):
        self.model = self.build_model(parameters)

    def build_model(self, p):
        model = Sequential()
        model.add(ZeroPadding2D(input_shape=p['input_shape']))
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Convolution2D(64, 3, 3))  # 64,7,7
        model.add(Flatten())  # 3136
        model.add(Dense(512))
        model.add(Dense())
        return model

    def act(self, observation):
        return 0

if __name__ == '__main__':
    p = dict(input_shape=(4, 84, 84))
    agent = DQN_Agent(p)
    print(agent.model.output_shape)

