# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dynamic_stitch

import tensorflow as tf


def f(a):
    pass


t = (1, 2)
p = (1, 2)

r = tf.dynamic_stitch(t, p)

f(r)
