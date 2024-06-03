# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/linspace#for_example

import tensorflow as tf


def f(a):
    pass


r = tf.linspace(10.0, 12.0, 3, name="linspace")

f(r)
