# Fixture for #1323: two relaxed functions, one call inside the other, each passing its default,
# so their deletions are disjoint ranges of one line.
import tensorflow as tf


def f(x, axis=-1):
    return tf.reduce_sum(x, axis=axis)


def g(x, scale=2):
    return x * scale


f(g(tf.constant([1.0, 2.0])))
