# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/sort#usage

import tensorflow as tf


def f(a):
    pass


a = [1, 10, 26.9, 2.8, 166.32, 62.3]
r = tf.sort(a)

f(r)
