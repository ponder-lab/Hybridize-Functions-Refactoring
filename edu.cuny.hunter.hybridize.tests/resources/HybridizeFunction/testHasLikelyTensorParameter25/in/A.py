import tensorflow as tf


def value_index(a, b):
    return a.value_index + b.value_index


# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/Graph#using_graphs_directly_deprecated
g = tf.Graph()
with g.as_default():
    # Defines operation and tensor in graph
    c = tf.constant(30.0)
    assert c.graph is g

a_val = tf.Tensor(g.get_operations()[0], 0, tf.float32)
b_val = tf.Tensor(g.get_operations()[0], 0, tf.float32)
assert a_val.dtype == tf.float32
assert a_val.shape == ()
assert b_val.dtype == tf.float32
assert b_val.shape == ()
result = value_index(a_val, b_val)
