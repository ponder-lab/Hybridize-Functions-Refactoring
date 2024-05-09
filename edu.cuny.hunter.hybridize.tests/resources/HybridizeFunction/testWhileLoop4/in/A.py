# https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/while_loop#example

import tensorflow as tf


def f(a):
    pass


i = [tf.constant(0)]
c = lambda i: tf.less(i, 10)
b = lambda i: (tf.add(i, 1),)
r = tf.while_loop(c, b, i)

f(r[0])
