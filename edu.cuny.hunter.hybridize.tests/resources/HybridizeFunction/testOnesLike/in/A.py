# From: https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/ones_like#for_example

import tensorflow as tf


def f(a):
    pass


tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
r = tf.ones_like(tensor)
f(r)
