import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


# Sample data
data = tf.constant([[-1, 3, 5, 2], [4, 0, -2, 7]])

# Find the top 2 values and their indices
values, indices = tf.math.top_k(data, k=2)

f(values)
g(indices)
