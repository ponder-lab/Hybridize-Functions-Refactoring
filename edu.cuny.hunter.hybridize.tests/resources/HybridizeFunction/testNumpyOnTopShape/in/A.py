import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def reduce_top(x, w):
    dims = get_shape(x)
    inner = np.prod(dims[-1:])
    flat = tf.reshape(x, [-1, inner])
    return tf.matmul(flat, w)


opaque = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]]))
w = tf.ones((2, 5))
reduce_top(opaque, w)
