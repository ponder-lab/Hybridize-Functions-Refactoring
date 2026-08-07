import numpy as np
import tensorflow as tf

rng = np.random

W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167])


def linear_regression(x):
    return W * x + b


# Eagerly the float64 NumPy argument is converted at the multiply under W's dtype; a bare
# @tf.function materializes it as a float64 tensor at the trace boundary instead, and the
# multiply raises the Mul dtype TypeError (#861 Case 1). A signature pinned to W's dtype
# reproduces the eager coercion at the boundary and runs.
linear_regression(X)
