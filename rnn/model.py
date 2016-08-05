import tensorflow as tf


class Model(object):

    def __init__(self, input_shape=None):

        self.initial_state = tf.placeholder(
            shape=input_shape, dtype=tf.float32)
        self.current_state = self.initial_state
        self.last_size = input_shape[-1]

    def add(self, layer):
        layer.incoming = self.current_state
        layer.input_size = self.last_size
        layer_output = layer.build_model()
        self.current_state = layer_output
        self.last_size = layer.hidden_dim

    def get_last_state(self):
        return self.current_state

    def feed(self, data):
        self.data = data

    def run(self):
        with tf.Session() as self.sess:
            self.sess.run(tf.initialize_all_variables())
            f = {self.initial_state: self.data}
            outputs = self.sess.run(self.current_state, f)
        return outputs
