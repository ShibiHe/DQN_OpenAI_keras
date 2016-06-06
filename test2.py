__author__ = 'frankhe'
from keras.models import Sequential, Model
from keras.layers import Convolution2D, ZeroPadding2D, Flatten, Dense, Activation, Input

x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
trainable_model.compile(optimizer='rmsprop', loss='mse')

# frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
# trainable_model.fit(data, labels)  # this updates the weights of `layer`

print frozen_model.trainable_weights, frozen_model.non_trainable_weights
print trainable_model.trainable_weights, trainable_model.non_trainable_weights
