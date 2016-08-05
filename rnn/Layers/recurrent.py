import tensorflow as tf
import math


class Recurrent(object):

    def __init__(self, hidden_dim):
        self.incoming = None
        self.hidden_dim = hidden_dim
        self.input_size = None
        self.initializer = tf.random_uniform_initializer(
            minval=-0.01, maxval=0.01, dtype=tf.float32)
        print('RNN initialized :', type(self).__name__)

    def create_variables(self):
        self.Wxh = tf.Variable(tf.random_normal(
            [self.input_size, self.hidden_dim],
            stddev=1.0 / math.sqrt(self.input_size)), dtype=tf.float32)
        self.Whh = tf.Variable(tf.random_normal(
            [self.hidden_dim, self.hidden_dim],
            stddev=1.0 / math.sqrt(self.hidden_dim)), dtype=tf.float32)

    def build_model(self):
        initial = tf.zeros(shape=[self.hidden_dim], dtype=tf.float32)
        states = tf.scan(self.recurrence, self.incoming, initializer=initial)
        return states

    def recurrence(self, prev, inp):
        print(type(self).__name__)
        i = tf.reshape(inp, shape=[1, -1])
        p = tf.reshape(prev, shape=[1, -1])
        h = tf.tanh((tf.matmul(p, self.Whh)) +
                    tf.matmul(i, self.Wxh))
        h = tf.reshape(h, [self.hidden_dim])
        return h
