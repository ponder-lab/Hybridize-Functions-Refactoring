# From https://www.tensorflow.org/guide/function#usage.

import tensorflow as tf


def add(t):
    return t[0] + t[1]


arg = (tf.ones([2, 2]), 1)  #  [[2., 2.], [2., 2.]]
assert type(arg) == tuple
result = add(arg)
