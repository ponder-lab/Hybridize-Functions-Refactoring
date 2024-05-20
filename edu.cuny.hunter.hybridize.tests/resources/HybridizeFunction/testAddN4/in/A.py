# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/add_n

import tensorflow as tf


def f(a):
    pass


a = [[3, 5], [4, 8]]
b = [[1, 6], [2, 9]]
r = tf.add_n([a, b, a])

f(r)
