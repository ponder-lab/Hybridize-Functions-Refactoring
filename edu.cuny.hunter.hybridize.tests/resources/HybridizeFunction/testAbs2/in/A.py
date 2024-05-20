# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dynamic_stitch

import tensorflow as tf


def f(a):
    pass


# real number
x = [-2.25, 3.25]
r = tf.abs(x)

f(r)
