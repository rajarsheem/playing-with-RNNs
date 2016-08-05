import tensorflow as tf


def index(state, idx):
    last_state = tf.expand_dims(tf.nn.embedding_lookup(
        state, tf.shape(state)[0] + idx), dim=0)
    return last_state


def zoneout(h, h_prev):
    assert h.get_shape() == h_prev.get_shape()
    r = tf.select(tf.random_uniform(h.get_shape()) >
                  0.7, tf.ones_like(h), tf.zeros_like(h))
    h_z = tf.mul(r, h_prev) + tf.mul(tf.sub(tf.ones_like(r), r), h)
    return h_z
