import tensorflow as tf


def index(state, idx):
    last_state = tf.expand_dims(tf.nn.embedding_lookup(
        state, tf.shape(state)[0] + idx), dim=0)
    return last_state
