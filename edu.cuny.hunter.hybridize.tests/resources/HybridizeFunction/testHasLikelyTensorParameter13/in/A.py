import tensorflow as tf


# @tf.function
def add(a, b):
  return a + b


# @tf.function
def dense_layer(x, w, b):
  return add(tf.matmul(x, w), b)


dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))
