import numpy as np
import tensorflow as tf


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [np.expand_dims(item, -1) if len(item.shape) == 1 else item for item in seq]
    return np.concatenate(seq, axis=-1)


t = tf.ones((3,))
r = cat([t, t, None])
assert r.shape == (3, 2)
