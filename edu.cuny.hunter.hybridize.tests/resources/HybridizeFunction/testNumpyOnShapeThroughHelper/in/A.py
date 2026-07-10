import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def reduce_via_shape(input_tensor, w):
    dims = get_shape(input_tensor)
    inner = np.prod(dims[-1:])
    flat = tf.reshape(input_tensor, [-1, inner])
    return tf.matmul(flat, w)


t = tf.ones((3, 4))
w = tf.ones((4, 5))
assert reduce_via_shape(t, w).shape == (3, 5)
