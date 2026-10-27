# Incomparable dtype with a non-tensor argument (#808). The decorator declares float32, but the call site passes a NumPy array of dtype
# int64, which TensorFlow silently casts to float32 at the signature boundary rather than rejecting. The signature is left unchanged,
# since changing its dtype would change the values the function computes.
import numpy as np
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(np.array(2))
