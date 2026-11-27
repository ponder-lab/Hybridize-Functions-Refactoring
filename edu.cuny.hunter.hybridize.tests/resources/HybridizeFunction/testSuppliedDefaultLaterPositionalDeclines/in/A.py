# Fixture for #1323: `a` is passed its default positionally, but `b` follows it, so deleting `a`
# alone would shift `b` into its place.
import tensorflow as tf


def f(x, a=1, b=2):
    return tf.reduce_sum(x) * a * b


f(tf.constant([1.0, 2.0]), 1, 2)
