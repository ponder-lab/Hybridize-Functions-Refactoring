# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/custom_gradient

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)

  def grad(upstream):
    return upstream * (1 - 1 / (1 + e))

  return tf.math.log(1 + e), grad


x = tf.constant(100.)
y = log1pexp(x)
dy_dx = tf.gradients(y, x)  # Will be NaN when evaluated.
