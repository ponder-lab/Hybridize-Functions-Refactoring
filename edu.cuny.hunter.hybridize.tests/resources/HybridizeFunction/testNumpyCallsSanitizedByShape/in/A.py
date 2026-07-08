import numpy as np
import tensorflow as tf


def encode(x):
    rows = x.shape[0]
    cols = x.shape[1]
    pos = np.zeros((rows, cols))
    pos[:, 0::2] = 1.0
    return x + tf.constant(pos, dtype=tf.float32)


t = tf.ones((2, 4))
r = encode(t)
assert r.shape == (2, 4)
assert float(r[0, 0]) == 2.0
assert float(r[0, 1]) == 1.0
