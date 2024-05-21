# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/random/shuffle

import tensorflow as tf


def f(a):
    pass


# real number
x = tf.constant([-2.25, 3.25])
r = tf.random.shuffle(x)

f(r)
