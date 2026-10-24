import tensorflow as tf


def trailing(x):
    return tf.shape(x)[1:]


t = tf.ones([2, 3, 4])
print(tf.reshape(t, tf.concat(([-1], trailing(t)), 0)).shape)
