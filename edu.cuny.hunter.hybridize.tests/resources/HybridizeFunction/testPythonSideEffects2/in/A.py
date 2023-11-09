# From https://www.tensorflow.org/guide/function#executing_python_side_effects.

import tensorflow as tf


def f(x):
  # print("Traced with", x) # This is a Python side-effect.
  tf.print("Executed with", x) # THis isn't.


f(1)
f(1)
f(2)
