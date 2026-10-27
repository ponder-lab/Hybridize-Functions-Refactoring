import tensorflow as tf


def flatten(x):
    x_shape = tf.shape(x)
    return tf.reshape(x, [-1]), x_shape


t = tf.ones([2, 3])
flat, shape = flatten(t)
print(tf.reshape(flat * 2.0, shape).shape)
