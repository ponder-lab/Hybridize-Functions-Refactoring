# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dynamic_stitch

import tensorflow as tf


def f(a):
    pass


r = tf.map_fn(fn=lambda t: tf.range(t, t + 3), elems=tf.constant([3, 5, 2]))

f(r)
