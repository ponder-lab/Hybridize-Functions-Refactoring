# From https://www.tensorflow.org/guide/function#usage.

import tensorflow as tf


@tf.function  # The decorator converts `add` into a `PolymorphicFunction`.
def add(a, b):
  return a + b


add(tf.ones([2, 2]), tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
