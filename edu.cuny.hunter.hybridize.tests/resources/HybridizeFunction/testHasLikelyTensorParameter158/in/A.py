# From https://www.tensorflow.org/guide/function#usage.

import tensorflow as tf


@tf.function
def add(t: tuple[tf.Tensor, tf.Tensor]):
    return t[0] + t[1]


arg = (tf.constant(2), tf.constant(2))
assert type(arg) == tuple
result = add(arg)
