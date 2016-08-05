from . import Recurrent
import tensorflow as tf
import math


class GRU(Recurrent):

    def __init__(self, hidden_dim, recurrent_layer_op=None):
        super().__init__(hidden_dim, recurrent_layer_op)

    def create_variables(self):
        super(GRU, self).create_variables()
        # weights associated with update gate
        self.Wxz = tf.Variable(tf.random_normal(shape=[
            self.input_size, self.hidden_dim],
            stddev=1.0 / math.sqrt(self.input_size)), dtype=tf.float32)
        self.Whz = tf.Variable(tf.random_normal(shape=[
            self.hidden_dim, self.hidden_dim],
            stddev=1.0 / math.sqrt(self.hidden_dim)), dtype=tf.float32)

        # weights associated with the reset gate
        self.Wxr = tf.Variable(tf.random_normal(shape=[
            self.input_size, self.hidden_dim],
            stddev=1.0 / math.sqrt(self.input_size)), dtype=tf.float32)
        self.Whr = tf.Variable(tf.random_normal(shape=[
            self.hidden_dim, self.hidden_dim],
            stddev=1.0 / math.sqrt(self.hidden_dim)), dtype=tf.float32)

    def build_model(self):
        return super(GRU, self).build_model()

    def recurrence(self, prev, inp):
        print(type(self).__name__)
        i = tf.reshape(inp, shape=[1, -1])
        p = tf.reshape(prev, shape=[1, -1])
        z = tf.nn.sigmoid(tf.matmul(i, self.Wxz) +
                          tf.matmul(p, self.Whz))    # update gate
        r = tf.nn.sigmoid(tf.matmul(i, self.Wxr) +
                          tf.matmul(p, self.Whr))    # reset gate
        h_ = tf.nn.tanh(tf.matmul(i, self.Wxh) +
                        tf.matmul(tf.mul(p, r), self.Whh))
        h = tf.mul(tf.sub(tf.ones_like(z), z), h_) + tf.mul(z, p)
        return tf.reshape(h, [self.hidden_dim])
