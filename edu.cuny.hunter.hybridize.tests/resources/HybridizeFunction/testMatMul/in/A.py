# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/linalg/matmul

import tensorflow as tf


def f(a):
    pass


a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
c = tf.matmul(a, b)

f(c)
