import collections

import tensorflow as tf

P = collections.namedtuple("P", ["a", "b"])


# Receives a member of each element of a dataset sliced from a namedtuple, read by name: an int32 vector.
def consume(b):
    return tf.reduce_sum(b)


floats = tf.ones((8, 4))
ints = tf.zeros((8, 3), dtype=tf.int32)
dataset = tf.data.Dataset.from_tensor_slices(P(floats, ints))

for e in dataset:
    consume(e.b)
