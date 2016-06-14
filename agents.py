__author__ = 'frankhe'
from keras.models import Sequential, Model
from keras.layers import Convolution2D, ZeroPadding2D, Flatten, Dense, Activation, Input, Lambda
from keras.optimizers import rmsprop, adam, sgd
from keras import backend as K
import theano.tensor as Tht
from theano.gradient import disconnected_grad
import numpy as np


class DqnAgent(object):
    def __init__(self, parameters):
        self.Q_model = None
        self.Q_old_model = None
        self.training_func = None
        self.Q_func = None
        self.build_model(parameters)

        self.states = []
        self.actions = []
        self.rewards = []

    @staticmethod
    def build_cnn_model(p, trainable=True):
        cnn_model = Sequential()
        cnn_model.add(ZeroPadding2D(input_shape=p['input_shape']))
        cnn_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), trainable=trainable))
        cnn_model.add(Activation('relu'))
        cnn_model.add(Convolution2D(64, 4, 4, subsample=(2, 2), trainable=trainable))
        cnn_model.add(Activation('relu'))
        cnn_model.add(Convolution2D(64, 3, 3, trainable=trainable))  # 64,7,7
        cnn_model.add(Activation('relu'))
        cnn_model.add(Flatten())  # 3136
        cnn_model.add(Dense(512, trainable=trainable))
        cnn_model.add(Activation('relu'))
        cnn_model.add(Dense(p['output_shape'], trainable=trainable))
        return cnn_model

    def build_model(self, p):
        S = Input(p['input_shape'], name='input_state')
        A = Input((1,), name='input_action', dtype='int32')
        R = Input((1,), name='input_reward')
        T = Input((1,), name='input_terminate', dtype='int32')
        NS = Input(p['input_shape'], name='input_next_sate')

        self.Q_model = self.build_cnn_model(p)
        self.Q_old_model = self.build_cnn_model(p, False)  # Q hat in paper
        self.Q_old_model.set_weights(self.Q_model.get_weights())  # Q' = Q

        Q_S = self.Q_model(S)  # batch * actions
        Q_NS = disconnected_grad(self.Q_old_model(NS))  # disconnected gradient is not necessary

        y = R + p['discount'] * (1-T) * K.max(Q_NS, axis=1, keepdims=True)  # batch * 1

        action_mask = K.equal(Tht.arange(p['num_actions']).reshape((1, -1)), A.reshape((-1, 1)))
        output = K.sum(Q_S * action_mask, axis=1).reshape((-1, 1))
        loss = K.sum((output - y) ** 2)  # sum could also be mean()

        optimizer = adam(p['learning_rate'])
        params = self.Q_model.trainable_weights
        update = optimizer.get_updates(params, [], loss)

        self.training_func = K.function([S, A, R, T, NS], loss, updates=update)
        self.Q_func = K.function([S], Q_S)

        # f = K0.function((S,), (Q_S,))
        # t = f([np.random.rand(10, *p['input_shape'])])
        # print(t[0])
        # a = np.random.randint(0, 4, (10,))
        # print(a)
        # print(t[0][range(10),a].shape)

        # f = K.function([NS, T, R], [y])
        # t = f([np.random.rand(10, *p['input_shape']), np.zeros((10, 1)), np.ones((10, 1))])
        # print(t[0])

        # f = K.function([S, A, R, T, NS], [y, output, loss])
        # s = np.random.rand(10, *p['input_shape'])
        # a = np.array([1, 0, 2, 3, 1, 0, 2, 3, 2, 1]).reshape((-1, 1))
        # ns = np.random.rand(10, *p['input_shape'])
        # r = np.ones((10, 1))
        # t = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 1]).reshape((-1, 1))
        # out = f([s, a, r, t, ns])
        # print(out[0])
        # print(out[1])
        # print(out[2])

    def update_Q_model(self):
        self.Q_old_model.set_weights(self.Q_model.get_weights)

    def act(self):
        pass

if __name__ == '__main__':
    param = dict(input_shape=(4, 84, 84), num_actions=5, output_shape=5, learning_rate = 0.01, discount=0.99)
    agent = DqnAgent(param)
    print(agent.Q_model.summary())

