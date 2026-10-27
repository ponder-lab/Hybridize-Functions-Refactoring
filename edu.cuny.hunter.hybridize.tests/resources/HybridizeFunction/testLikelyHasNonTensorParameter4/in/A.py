import tensorflow as tf


def f(a, b):
    pass


def g(a, b):
    pass


f(tf.constant(5), 5)
g(tf.constant(5), 5)
g(tf.constant(5), 6)
