# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dynamic_stitch

import tensorflow as tf


def f(a):
    pass


t = (tf.constant(1), tf.constant(2))
p = (tf.constant(1), tf.constant(2))

r = tf.dynamic_stitch(t, p)

f(r)