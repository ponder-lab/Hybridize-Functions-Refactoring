# From: https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dynamic_partition

import tensorflow as tf


def f(a):
    pass


t = tf.constant(1)
p = tf.constant(1)

r = tf.dynamic_partition(t, p, 2)

f(r)
