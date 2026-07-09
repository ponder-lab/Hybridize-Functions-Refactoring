import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def einsum_via_matmul(input_tensor, w, num_inner_dims):
    input_shape = get_shape(input_tensor)
    inner_dims = input_shape[-num_inner_dims:]
    inner = np.prod(inner_dims)
    flat = tf.reshape(input_tensor, [-1, inner])
    return tf.matmul(flat, w)


def reduce_tail(x, w):
    return einsum_via_matmul(x, w, 1)


t = tf.ones((3, 4))
w = tf.ones((4, 5))
assert reduce_tail(t, w).shape[1] == 5
