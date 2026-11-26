# Fixture for #1323: `axis` is defaulted and non-tensor, and one call passes it, but only as its
# default, so deleting that argument changes nothing and `axis` can be left out of the signature.
import tensorflow as tf


def f(x, axis=-1):
    return tf.reduce_sum(x, axis=axis)


f(tf.constant([1.0, 2.0]), axis=-1)
f(tf.constant([3.0, 4.0]))
