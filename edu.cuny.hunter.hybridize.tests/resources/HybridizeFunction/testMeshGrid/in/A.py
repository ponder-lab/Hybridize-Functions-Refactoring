# https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/meshgrid#examples

import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def h(a):
    pass


x = [1, 2, 3]
y = [4, 5, 6]

f(tf.meshgrid(x, y))

X, Y = tf.meshgrid(x, y)

g(X)
h(X)
