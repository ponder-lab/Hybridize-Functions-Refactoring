import tensorflow as tf


def f(x, axis=-1):
    return tf.reduce_sum(x, axis=axis)
