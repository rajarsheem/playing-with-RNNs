from rnn import Model
from rnn.utils.ops import zoneout
from rnn.layers import Recurrent, GRU
from rnn.utils.ops import index
from rnn.layers import Loss
import tensorflow as tf
import numpy as np


def sine_data(ix, size=50):
    x = np.arange(ix, ix + size, step=0.2)
    y = np.cos(x)
    return y[:-1], y[1:]

with tf.Graph().as_default():
    # no. of time steps is 30, each time step has input of size 5
    # in case of dynamic recurrence, leave no. of time steps as None
    model = Model(input_shape=[30, 1], output_shape=[30, 1])
    model.add(Recurrent(10))
    model.add(GRU(15))
    state = model.get_last_state()
    loss = Loss(state)
    print(state2)
    # last_step = index(state, -1)
    # print(state)

    # feeding data to model
    # shape of the data must be of same size you provided in Model()
    model.feed(np.random.randn(30, 1))
    print(model.run(state).shape)
