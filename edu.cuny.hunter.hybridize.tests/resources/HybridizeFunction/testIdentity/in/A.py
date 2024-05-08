# https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/identity#for_example

import tensorflow as tf


def f(a):
    pass


a = tf.constant([0.78])
a_identity = tf.identity(a)
f(a_identity)
