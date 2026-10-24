import tensorflow as tf


# Receives the first grid `tf.meshgrid` returns, unpacked.
def unpacked(grid):
    return tf.reduce_sum(grid)


# Receives the first grid `tf.meshgrid` returns, subscripted.
def subscripted(grid):
    return tf.reduce_sum(grid)


# Receives the list of grids `tf.meshgrid` returns.
def grids(pair):
    return tf.reduce_sum(pair[0]) + tf.reduce_sum(pair[1])


x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([4.0, 5.0])

X, Y = tf.meshgrid(x, y)
unpacked(X)

subscripted(tf.meshgrid(x, y)[0])

grids(tf.meshgrid(x, y))
