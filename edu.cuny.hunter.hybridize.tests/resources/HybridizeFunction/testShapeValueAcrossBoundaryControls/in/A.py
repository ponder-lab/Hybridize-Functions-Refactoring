import tensorflow as tf


def constant_reshape(x):
    return tf.reshape(x, [3, 2])


def flatten_batch(x):
    x_shape = tf.shape(x)
    return tf.reshape(x, [x_shape[0], -1])


def scale(x):
    return x * 2.0 + tf.reduce_sum(x)


t = tf.ones([2, 3])
print(constant_reshape(t).shape)
print(flatten_batch(t).shape)
print(scale(t).shape)
