# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/einsum

import tensorflow as tf


def f(a):
    pass


u = tf.random.normal(shape=[5])
v = tf.random.normal(shape=[5])
e = tf.einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]

f(e)
