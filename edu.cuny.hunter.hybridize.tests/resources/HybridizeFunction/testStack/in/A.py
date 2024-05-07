# From: https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/stack#for_example

import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])

r = tf.stack([x, y, z])
f(r)

s = tf.stack([x, y, z], axis=1)
g(s)
