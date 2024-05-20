# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dynamic_stitch

import tensorflow as tf


def f(a):
    pass


t = [[1.5, 2.0], [3.2, 4.8]]

r = tf.linalg.eigh(t)

f(r)
