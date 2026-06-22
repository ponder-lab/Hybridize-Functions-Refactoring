import tensorflow as tf


def f(x, *, y: tf.Tensor):
    return y


f(tf.constant(1), y=tf.ones([2, 3]))
