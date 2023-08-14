import tensorflow as tf


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32), tf.TensorSpec(shape=[None], dtype=tf.int32)))
def add(a, b):
  return a + b


a = tf.range(3, 18, 3)
b = tf.range(5)
c = add(a, b)
