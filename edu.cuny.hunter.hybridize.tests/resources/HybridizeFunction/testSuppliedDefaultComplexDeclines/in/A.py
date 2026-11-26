# Fixture for #1323: the default is a complex literal, which is not among the literals compared,
# so the argument passing it is not deleted.
import tensorflow as tf


def f(x, c=1j):
    return tf.reduce_sum(x) * abs(c)


f(tf.constant([1.0, 2.0]), c=1j)
