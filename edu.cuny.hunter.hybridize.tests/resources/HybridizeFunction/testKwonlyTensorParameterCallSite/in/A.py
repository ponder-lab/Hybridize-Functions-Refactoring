import tensorflow as tf


def f(x, *, y):
    return y


f(tf.constant(1), y=tf.ones([2, 3]))
