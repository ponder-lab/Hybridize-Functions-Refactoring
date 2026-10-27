import numpy as np
import tensorflow as tf


# Receives an `np.full` array of int32 indices.
def gather(x, indices):
    return tf.gather(x, indices)


gather(tf.constant([1.0, 2.0, 3.0]), np.full((2,), 1, dtype=np.int32))
