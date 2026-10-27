import tensorflow as tf


def restore(x, shape):
    return tf.reshape(x, shape)


t = tf.ones([2, 6])
print(restore(t, tf.shape(t)[::-1]).shape)
