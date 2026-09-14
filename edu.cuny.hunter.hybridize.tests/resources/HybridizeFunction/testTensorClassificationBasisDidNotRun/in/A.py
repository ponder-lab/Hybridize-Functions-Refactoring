import tensorflow as tf


def scale(x, factor):
    return x * factor


def never_called(z):
    return z + 1


t = tf.ones((2, 2))
scale(t, 3)
