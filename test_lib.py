from rnn import Model
from rnn.utils.ops import zoneout
from rnn.layers import Recurrent, GRU
from rnn.utils.ops import index
import tensorflow as tf
import numpy as np

with tf.Graph().as_default():
    # no. of time steps is 30, each time step has input of size 5
    # in case of dynamic recurrence, leave no. of time steps as None
    # (i.e. input_shape = [None, 5])
    model = Model(input_shape=[30, 5])

    model.add(Recurrent(15))
    model.add(GRU(15))
    state = model.get_last_state()
    last_step = index(state, -1)
    # print(state)

    # feeding data to model
    # shape of the data must be of same size you provided in Model()
    # i.e. [30, 5] in our case
    model.feed(np.random.randn(30, 5))
    print(model.run().shape)
