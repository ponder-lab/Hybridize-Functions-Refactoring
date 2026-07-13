import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def head_over_dynamic(x, w):
    dims = get_shape(x)
    inner = np.prod(dims[:1])
    flat = tf.reshape(x, [-1, inner])
    return tf.matmul(flat, w)


# `x`'s leading (batch) axis is graph-time `None` (dynamic evidence via
# `tf.keras.Input`), so `get_shape(x)[:1]` covers a provably-dynamic dimension
# and numpy (`np.prod`) over it consumes it. A compile-time-constant source would
# instead let Ariadne fold the leading extent to a static size (wala/ML#722), so
# the earlier `tf.reshape(constant, [-1, 5])` feed no longer suffices.
inp = tf.keras.Input(shape=(5,))
w = tf.ones((5, 2))
head_over_dynamic(inp, w)
