import tensorflow as tf


def add(a, b):
    return a + b


a_val = tf.Variable([1.0, 2.0])
b_val = tf.Variable([2.0, 2.0])
assert a_val.dtype == tf.float32
assert a_val.shape == (2,)
assert b_val.dtype == tf.float32
assert b_val.shape == (2,)
c = add(a_val, b_val)
