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


# Reshaping a ⊤-shaped tensor with an explicit trailing extent yields a dynamic
# leading dimension and a static trailing one, so the shape slice passed into
# `prod_of` covers a provably-dynamic dimension.
dyn = tf.reshape(
    tf.constant(np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])),
    [-1, 5],
)
w = tf.ones((2, 3), dtype=tf.float64)
assert head_via_arg(dyn, w).shape == (5, 3)
