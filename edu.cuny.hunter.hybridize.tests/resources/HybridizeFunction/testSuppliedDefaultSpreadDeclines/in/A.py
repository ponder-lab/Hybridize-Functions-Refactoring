# Fixture for #1323: one call spreads its arguments, so which of them fills `axis` is not
# written at the call.
import tensorflow as tf


def f(x, axis=-1):
    return tf.reduce_sum(x, axis=axis)


arguments = (tf.constant([1.0, 2.0]), -1)
f(*arguments)
f(tf.constant([3.0, 4.0]), axis=-1)
