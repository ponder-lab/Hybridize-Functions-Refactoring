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


# Reshaping a ⊤-shaped tensor with an explicit trailing extent yields a dynamic
# leading dimension and a static trailing one, so numpy over the leading
# dimension extracted through the BERT-style `get_shape_list` consumes a
# provably-dynamic dimension.
dyn = tf.reshape(
    tf.constant(np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])),
    [-1, 5],
)
w = tf.ones((2, 3), dtype=tf.float64)
assert head_via_list(dyn, w).shape == (5, 3)
