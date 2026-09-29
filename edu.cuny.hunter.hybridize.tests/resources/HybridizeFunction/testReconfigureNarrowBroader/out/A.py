# Narrowing path (#808), supplied-broader case. The decorator's shape=None admits any shape, broader than the concrete scalar the
# call site requires. Under the closed-world assumption the reachable call sites are all the callers, and each conforms to the
# inferred signature, so the supplied one is narrowed to it.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
