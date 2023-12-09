# From https://www.tensorflow.org/guide/function#usage.

import tensorflow as tf


@tf.function  # The decorator converts `add` into a `PolymorphicFunction`.
def add(t):
  return t[0][0] + t[0][1] + t[1][0] + t[1][0]


arg = ((tf.ones([2, 2]), tf.ones([2, 2])), (tf.ones([2, 2]), tf.ones([2, 2])))
assert type(arg) == tuple
assert type(arg[0]) == tuple
assert type(arg[1]) == tuple
result = add(arg)
