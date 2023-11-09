# From https://www.tensorflow.org/guide/function#executing_python_side_effects.

import tensorflow as tf


def f(x):
    # Assigning x to a."
    a = x
    print("Traced with", a)  # This is a transitive Python side-effect.
    tf.print("Executed with", x)  # This isn't.


f(1)
f(1)
f(2)
