# From https://www.tensorflow.org/guide/function#usage.

import tensorflow as tf


@tf.function  # The decorator converts `add` into a `PolymorphicFunction`.
def add(t):
  return t[0] + t[1]


arg = (1, tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
assert type(arg) == tuple
result = add(arg)
