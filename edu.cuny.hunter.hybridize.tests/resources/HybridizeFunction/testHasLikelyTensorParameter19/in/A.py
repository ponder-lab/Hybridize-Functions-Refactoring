import tensorflow as tf


def add(z, a, b):
  return a + b


c = add(5, tf.ones([1, 2]), tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
