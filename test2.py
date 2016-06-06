__author__ = 'frankhe'
from keras.models import Sequential, Model
from keras.layers import Convolution2D, ZeroPadding2D, Flatten, Dense, Activation, Input
from keras.optimizers import rmsprop, adam, sgd
from keras import backend as K

x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')
print layer.trainable_weights, frozen_model.non_trainable_weights

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

# frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
# trainable_model.fit(data, labels)  # this updates the weights of `layer`
