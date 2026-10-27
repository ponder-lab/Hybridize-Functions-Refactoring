# A broader supplied signature on a function in a program that saves an array with NumPy (#808). Saving an array exports no TensorFlow
# interface, so the closed-world narrowing still applies.
import numpy as np
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return t + 1


if __name__ == "__main__":
    f(tf.constant(2.0))
    np.save("saved.npy", np.ones(1))
