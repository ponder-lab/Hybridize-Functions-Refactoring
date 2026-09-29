# Narrowing path (#808) for a spec whose shape and dtype are passed positionally. The narrowed signature spells both by keyword.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
