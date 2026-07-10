import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def prod_of(dims):
    return np.prod(dims)


def head_or_tail(tensor, flag):
    if flag:
        return get_shape(tensor)[:1]
    return get_shape(tensor)[-1:]


def tail_of_pair(tensor):
    return get_shape(tensor)[-2:][-1:]


def reduce_merged(x, w):
    picked = prod_of(head_or_tail(x, True))
    inner = np.prod(tail_of_pair(x))
    flat = tf.reshape(x, [picked, inner])
    return tf.matmul(flat, w)


t = tf.ones((3, 4))
w = tf.ones((4, 5))
assert reduce_merged(t, w).shape == (3, 5)
