# Keyword variant of the NumPy-argument dtype disagreement (#808). The call site passes a NumPy array of dtype int64 by keyword, which
# TensorFlow silently casts to the declared float32. The signature is left unchanged.
import numpy as np
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(t=np.array(2))
