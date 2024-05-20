# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/equal#for_example

import tensorflow as tf


def f(a):
    pass


x = tf.constant([2, 4])
y = tf.constant(2)
r = tf.math.equal(x, y)

f(r)
