# From https://www.tensorflow.org/guide/function#executing_python_side_effects.

import tensorflow as tf


def f(x):
  print("Traced with", x)
  tf.print("Executed with", x)


f(1)
