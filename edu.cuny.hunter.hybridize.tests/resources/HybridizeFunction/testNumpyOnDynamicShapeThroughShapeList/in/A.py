import numpy as np
import tensorflow as tf


def get_shape_list(tensor):
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for index, dim in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def head_via_list(x, w):
    inner = np.prod(get_shape_list(x)[:1])
    flat = tf.reshape(x, [-1, inner])
    return tf.matmul(flat, w)


# `x`'s leading axis is graph-time `None` (dynamic evidence via `tf.keras.Input`),
# so the BERT-style `get_shape_list` patches it with `tf.shape(...)` and numpy over
# the leading dimension consumes a provably-dynamic dimension. A compile-time-constant
# source would let Ariadne fold it (wala/ML#722).
inp = tf.keras.Input(shape=(5,))
w = tf.ones((2, 3), dtype=tf.float64)
head_via_list(inp, w)
