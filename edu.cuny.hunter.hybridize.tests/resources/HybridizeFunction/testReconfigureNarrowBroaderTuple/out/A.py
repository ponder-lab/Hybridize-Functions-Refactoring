# Tuple-literal variant of the narrowing path (#808). The supplied signature is a parenthesized tuple whose single spec (shape=None)
# is broader than the concrete scalar the call site requires, so the whole tuple is replaced by the inferred signature.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
