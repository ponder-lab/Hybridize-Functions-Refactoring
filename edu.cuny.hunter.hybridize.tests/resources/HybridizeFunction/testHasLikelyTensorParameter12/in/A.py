import tensorflow as tf

# Can benefit from tf.function per the docs.
def add(a, b):
  return a + b


v = tf.Variable(1.0)
with tf.GradientTape() as tape:
  result = add(v, 1.0) # Using tf.Variable is kind of like using tf.Tensor. Is 1.0 a tensor?
tape.gradient(result, v)

# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/Variable:
# Just like any Tensor, variables created with Variable() can be used as inputs to operations.
# Additionally, all the operators overloaded for the Tensor class are carried over to variables.
