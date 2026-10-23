import numpy as np
import tensorflow as tf


# Receives an int64 array cast to another array's float32 dtype as `y`.
def add(x, y):
    return x + y


x = np.ones((2, 3), dtype=np.float32)
w = np.ones((2, 3), dtype=np.int64)
add(tf.constant(x), w.astype(x.dtype))
