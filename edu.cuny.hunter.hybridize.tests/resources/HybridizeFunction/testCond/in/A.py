# https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/cond#example

import tensorflow as tf


def f(a):
    pass


x = tf.constant(2)
y = tf.constant(5)


def f1(): return tf.multiply(x, 17)


def f2(): return tf.add(y, 23)


r = tf.cond(tf.less(x, y), f1, f2)
# r is set to f1().
# Operations in f2 (e.g., tf.add) are not executed.

f(r)
