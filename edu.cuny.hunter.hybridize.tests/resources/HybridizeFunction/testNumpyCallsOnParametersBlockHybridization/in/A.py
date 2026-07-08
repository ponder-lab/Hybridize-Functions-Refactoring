import numpy as np
import tensorflow as tf


def scale(x):
    return np.maximum(x, 0.0)


def helper(y):
    return np.sum(y)


def outer(x):
    return helper(x)


def half(z):
    return z * 0.5


def chained(x):
    h = half(x)
    return np.abs(h)


def compute(x):
    return tf.reduce_mean(x)


t = tf.constant([1.0, -2.0, 3.0])
assert scale(t)[1] == 0.0
assert float(outer(t)) == 2.0
assert float(chained(t)[1]) == 1.0
assert abs(float(compute(t)) - 2.0 / 3.0) < 1e-6
