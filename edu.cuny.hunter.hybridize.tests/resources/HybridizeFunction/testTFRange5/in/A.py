# From: https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/range#for_example

import tensorflow as tf


def f(a):
    pass


r = [tf.constant(3), tf.constant(6)]

for i in r:
    f(i)
