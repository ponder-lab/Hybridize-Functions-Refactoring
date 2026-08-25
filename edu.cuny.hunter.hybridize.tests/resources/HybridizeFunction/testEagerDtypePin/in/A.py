import numpy as np
import tensorflow as tf

V = tf.Variable(tf.zeros([2]), name="scale_weight")
M = tf.Variable(tf.zeros([2, 2]), name="matmul_weight")

X = np.array([1.5, 2.5])


def scale(x):
    return tf.reduce_sum(V * x)


def matmuled(x):
    return tf.reduce_sum(tf.matmul(M, x))


def multiplied(x):
    return tf.reduce_sum(tf.multiply(V, x))


# Eagerly the float64 NumPy argument is converted at the multiply under V's float32; a
# bare decorator would materialize it as float64 and raise at the multiply, while a
# signature pinned to float32 reproduces the eager coercion at the boundary (#861 Case 1).
scale(X)


# The same coercion through an operation whose dtype rule the type analysis does not
# declare, so the reported dtype is the fed one and the repair is still this tool's.
matmuled(np.array([[1.5, 2.5], [3.5, 4.5]]))

multiplied(X)
