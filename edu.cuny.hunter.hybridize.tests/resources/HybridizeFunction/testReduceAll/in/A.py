# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/reduce_all#for-example

import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def h(a):
    pass


x = tf.constant([[True, True], [False, False]])

r = tf.math.reduce_all(x)
f(r)

r = tf.math.reduce_all(x, 0)
g(r)

r = tf.math.reduce_all(x, 1)
h(r)
