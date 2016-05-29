import numpy as np
import tensorflow as tf
import data
from tensorflow.models.rnn.ptb import reader

def input_target_generator(min_duration=5, max_duration=50):
    duration = np.random.randint(min_duration, max_duration)
    inputs = np.random.randn(duration).astype(np.float32)
    targets = np.cumsum(inputs).astype(np.float32)
    return inputs.reshape(-1, 1), targets.reshape(-1, 1)

def vanilla_rnn_step(h_prev, x):
    h_prev = tf.reshape(h_prev, [1, hidden_layer_size])
    x = tf.reshape(x, [1, input_size])
    print('hello')
    with tf.variable_scope('rnn_block'):
        # U_z = tf.get_variable('U_z',shape=[input_size, hidden_layer_size])
        # W_z = tf.get_variable('W_z',shape=[hidden_layer_size, hidden_layer_size])
        # z = tf.sigmoid(tf.matmul(x, U_z) + tf.matmul(h_prev, W_z))
        # U_r = tf.get_variable('U_r', shape=[input_size, hidden_layer_size])
        # W_r = tf.get_variable('W_r', shape=[hidden_layer_size, hidden_layer_size])
        # r = tf.sigmoid(tf.matmul(x, U_r) + tf.matmul(h_prev, W_r))
        # W_h = tf.get_variable(
        #     'W_h', shape=[hidden_layer_size, hidden_layer_size])
        # W_x = tf.get_variable(
        #     'W_x', shape=[input_size, hidden_layer_size])
        # b = tf.get_variable('b', shape=[hidden_layer_size],
        #                     initializer=tf.constant_initializer(0.0))
        # h_ = tf.tanh(tf.matmul(tf.mul(r, h_prev), W_h) + tf.matmul(x, W_x))
        # h = tf.mul((1 - z), h_)  + tf.mul(z, h_prev)
        # h = tf.reshape(h, [hidden_layer_size], name='h')

        W_h = tf.get_variable('W_h', shape=[hidden_layer_size, hidden_layer_size])
        W_x = tf.get_variable('W_x', shape=[input_size, hidden_layer_size])
        b = tf.get_variable('b', shape=[hidden_layer_size], initializer=tf.constant_initializer(0.0))
        h =  tf.matmul(h_prev, W_h) + tf.matmul(x, W_x) + b
        h = tf.reshape(h, [hidden_layer_size], name='h')
    return h

def compute_predictions(inputs):
    with tf.variable_scope('states'):
        initial_state = tf.zeros([hidden_layer_size],
                                 name='initial_state')
        states = tf.scan(vanilla_rnn_step, inputs,
                                     initializer=initial_state, name='states')

    with tf.variable_scope('predictions'):
        W_pred = tf.get_variable(
            'W_pred', shape=[hidden_layer_size, target_size])
        b_pred = tf.get_variable('b_pred', shape=[target_size],
                                 initializer=tf.constant_initializer(0.0))
        predictions = tf.add(tf.matmul(states, W_pred), b_pred, name='predictions')

    return states, predictions

def compute_loss(targets, predictions):
    with tf.variable_scope('loss'):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, targets), name='loss')
        # loss = tf.reduce_mean((targets - predictions)**2, name='loss')
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(targets, predictions))))
        return loss

input_size = 5
target_size = 5
hidden_layer_size = 300
num_samples = 10

inputs = tf.placeholder(tf.float32, shape=[num_samples, input_size])
targets = tf.placeholder(tf.float32, shape=[num_samples, target_size])

initializer = tf.random_uniform_initializer(-0.1, 0.1)
with tf.variable_scope('model', initializer=initializer):
    states, predictions = compute_predictions(inputs)
    loss = compute_loss(targets, predictions)

initial_learning_rate=0.001
num_steps_per_decay=10000
decay_rate=0.1
max_global_norm=1.0

trainables = tf.trainable_variables()
grads = tf.gradients(loss, trainables)
grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
grad_var_pairs = zip(grads, trainables)

# global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
# learning_rate = tf.train.exponential_decay(
#     initial_learning_rate, global_step, num_steps_per_decay,
#     decay_rate, staircase=True)
optimizer = tf.train.AdagradOptimizer(initial_learning_rate)
optimize_op = optimizer.apply_gradients(grad_var_pairs)



# d = data.get_data()[1]
data = reader.ptb_raw_data('/home/rajarshee/Documents/Data/simple-examples/data')[1]

sess = tf.Session()


# ema = tf.train.ExponentialMovingAverage(decay=0.5)
# update_loss_ema = ema.apply([loss])
# loss_ema = ema.average(loss)
sess.run(tf.initialize_all_variables())

for step, (a, b) in enumerate(reader.ptb_iterator(data, num_samples, input_size)):
    # a, b = input_target_generator(num_samples,num_samples + 1)
    loss_ema_, _ = sess.run(
        [loss, optimize_op],
        {inputs: a, targets: b})
    if step % 10 == 0:
        print('\rStep %d. Loss EMA: %.6f.' % (step+1, loss_ema_))



# a, b = input_target_generator(10,11)
# pred = sess.run(predictions, {inputs: a})
# for x, y in zip(pred, b):
#     print(x, y)
