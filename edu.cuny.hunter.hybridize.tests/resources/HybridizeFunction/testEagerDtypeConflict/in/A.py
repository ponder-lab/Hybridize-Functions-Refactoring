import numpy as np
import tensorflow as tf

V32 = tf.Variable(tf.zeros([2]), name="v32")
V64 = tf.Variable(tf.zeros([2], dtype=tf.float64), name="v64")

X = np.array([1.5, 2.5])


def combine(x):
    return tf.reduce_sum(V32 * x), tf.reduce_sum(V64 * x)


# Eagerly the NumPy argument is coerced per operation: float32 at the V32 multiply and
# float64 at the V64 multiply, so both succeed. Any single input signature breaks one of
# the two, so hybridization must decline (#861 Case 1, the plural fallback).
combine(X)
