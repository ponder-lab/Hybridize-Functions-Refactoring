import tensorflow as tf


def add(a, b):
    return tf.sparse.add(a, b)


a_val = tf.SparseTensor([[0, 0], [1, 2]], [1, 2], [3, 4])
b_val = tf.SparseTensor([[0, 0], [1, 2]], [1, 2], [3, 4])
assert a_val.dtype == tf.int32
assert a_val.shape == (3, 4)
assert b_val.dtype == tf.int32
assert b_val.shape == (3, 4)
c = add(a_val, b_val)
