# Modify path (#596), supplied-broader case. The decorator's shape=None admits any shape, broader than the concrete scalar the call
# site requires. The broader signature may be intentional, so it is preserved (informational), not overwritten.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
