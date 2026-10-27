# Fixture for a tuple display: its length and the type at each position are known, so the
# parameter reduces to a nested specification with one entry per position.
import tensorflow as tf


def f(xs):
    return xs[0]


f((tf.ones([2]), tf.ones([3], dtype=tf.int32)))
