# Modify path (#596), flag-off guard. With input-signature inference disabled (the suite default), an existing signature is never
# compared or overwritten; the precondition matrix is unchanged.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
