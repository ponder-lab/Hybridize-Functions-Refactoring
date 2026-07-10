import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def prod_of(dims):
    return np.prod(dims)


def via_arg(x, w):
    n = prod_of(get_shape(x))
    return tf.reshape(x, [-1, n])


t = tf.ones((3, 4))
w = tf.ones((4, 5))
via_arg(t, w)
