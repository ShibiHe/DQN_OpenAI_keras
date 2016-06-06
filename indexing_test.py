from __future__ import print_function
__author__ = 'frankhe'
import theano.tensor as T
from theano.compile import function
import numpy as np

num_actions = 3
batch_size = None

q_s = T.matrix()
a = T.col()

# batch_size = q_s.shape[0]
# out = q_s[range(batch_size), a.reshape(batch_size)]

mask = T.eq(T.arange(num_actions).reshape((1, -1)), a.reshape((-1, 1)))
out_t = q_s * mask
out = T.sum(out_t, axis=1, keepdims=True)

f = function([q_s, a], out)

q_s_ = np.random.rand(5, num_actions)
a_ = np.array([1, 0, 2, 1, 2]).reshape((5, 1))
print(f(q_s_, a_))
