# Fixture for #1323: each call passes a value Python's `==` equates with the default, but of
# another kind or sign, so the argument is not deleted: `1.0` for `1`, `1` for `True`, and
# `-0.0` for `0.0`.
import tensorflow as tf


def f(x, n=1):
    return tf.reduce_sum(x) * n


def g(x, b=True):
    return tf.reduce_sum(x) if b else x


def h(x, z=0.0):
    return tf.reduce_sum(x) + z


f(tf.constant([1.0, 2.0]), n=1.0)
g(tf.constant([1.0, 2.0]), b=1)
h(tf.constant([1.0, 2.0]), z=-0.0)
