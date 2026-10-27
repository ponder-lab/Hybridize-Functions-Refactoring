import tensorflow as tf


# Receives a scalar built from Python's `float` as `factor`.
def scale(x, factor):
    return x * factor


n = 3
scale(tf.ones(2), tf.constant(float(n)))
