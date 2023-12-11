# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/function#retracing.

import tensorflow as tf


@tf.function
def f(x, y):
  return tf.abs(x)


f1 = f.get_concrete_function(1, 2)
f2 = f.get_concrete_function(2, 3)  # Slow - compiles new graph
assert (f1 is f2) == False

f1 = f.get_concrete_function(tf.constant(1), 2)
f2 = f.get_concrete_function(tf.constant(2), 2)  # Fast - reuses f1
assert (f1 is f2) == True
