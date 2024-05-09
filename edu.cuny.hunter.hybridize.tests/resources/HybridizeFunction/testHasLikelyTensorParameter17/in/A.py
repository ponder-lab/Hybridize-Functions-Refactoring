import tensorflow as tf


# From https://www.tensorflow.org/guide/function#executing_python_side_effects.
# @tf.function
def f(x):
    print("Traced with", x)
    tf.print("Executed with", x)


f(1)
f(1)
f(2)
