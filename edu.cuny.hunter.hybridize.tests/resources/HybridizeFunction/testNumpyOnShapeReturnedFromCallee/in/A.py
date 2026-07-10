import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def tail_dims(tensor):
    return get_shape(tensor)[-1:]


def reduce_returned_tail(x, w):
    inner = np.prod(tail_dims(x))
    flat = tf.reshape(x, [-1, inner])
    return tf.matmul(flat, w)


# The leading dimension is dynamic and the trailing one static: the descriptor
# `tail_dims` returns covers only the trailing dimension, so numpy over it is
# graph-safe, while the all-dimensions extractor fallback would also cover the
# dynamic leading dimension and wrongly decline.
dyn = tf.reshape(
    tf.constant(np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])),
    [-1, 5],
)
w = tf.ones((5, 3), dtype=tf.float64)
assert reduce_returned_tail(dyn, w).shape == (2, 3)
