import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def prod_of(dims):
    return np.prod(dims)


def head_via_arg(x, w):
    inner = prod_of(get_shape(x)[:1])
    flat = tf.reshape(x, [-1, inner])
    return tf.matmul(flat, w)


# `x`'s leading axis is graph-time `None` (dynamic evidence via `tf.keras.Input`),
# so the shape slice passed into `prod_of`'s numpy sink covers a provably-dynamic
# dimension. A compile-time-constant source would let Ariadne fold it (wala/ML#722).
inp = tf.keras.Input(shape=(5,))
w = tf.ones((2, 3), dtype=tf.float64)
head_via_arg(inp, w)
