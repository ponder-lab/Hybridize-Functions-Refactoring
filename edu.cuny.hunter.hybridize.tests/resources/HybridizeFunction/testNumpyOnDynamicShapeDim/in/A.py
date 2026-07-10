import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def head_over_dynamic(x, w):
    dims = get_shape(x)
    inner = np.prod(dims[:1])
    flat = tf.reshape(x, [-1, inner])
    return tf.matmul(flat, w)


# Reshaping a ⊤-shaped tensor with an explicit trailing extent yields a dynamic
# leading dimension and a static trailing one, so numpy over the leading
# dimension consumes a provably-dynamic dimension.
dyn = tf.reshape(
    tf.constant(np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])),
    [-1, 5],
)
w = tf.ones((5, 2))
head_over_dynamic(dyn, w)
