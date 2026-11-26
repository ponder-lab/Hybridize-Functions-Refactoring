# Fixture for #1323: the default is passed as the call's last positional argument.
import tensorflow as tf


def f(x, axis=-1):
    return tf.reduce_sum(x, axis=axis)


f(tf.constant([1.0, 2.0]), -1)
f(tf.constant([3.0, 4.0]))
