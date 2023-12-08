# From https://www.tensorflow.org/guide/function#usage.

import tensorflow as tf


@tf.function  # The decorator converts `add` into a `PolymorphicFunction`.
def add(t):
  return t[0] + t[1]


result = add((tf.ones([2, 2]), tf.ones([2, 2])))  #  [[2., 2.], [2., 2.]]
