# Modify path (#596), agreement case. The supplied signature matches the inferred one exactly, so the refactoring is a no-op.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
