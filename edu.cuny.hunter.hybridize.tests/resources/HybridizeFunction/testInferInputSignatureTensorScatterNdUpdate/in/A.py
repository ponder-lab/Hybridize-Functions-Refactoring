import tensorflow as tf


# Receives a float32 tensor `tf.tensor_scatter_nd_update` returns.
def total(x):
    return tf.reduce_sum(x)


# Receives an int32 tensor `tf.tensor_scatter_nd_update` returns.
def int_total(x):
    return tf.reduce_sum(x)


updated = tf.tensor_scatter_nd_update(tf.zeros((4,)), [[1], [3]], [1.0, 2.0])
total(updated)

int_updated = tf.tensor_scatter_nd_update(tf.zeros((6,), dtype=tf.int32), [[0]], [5])
int_total(int_updated)
