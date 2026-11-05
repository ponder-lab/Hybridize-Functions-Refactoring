import collections

import tensorflow as tf

P = collections.namedtuple("P", ["a", "b"])


# Receives the members of each element of a zip whose first member is a namedtuple of datasets, unpacked by position: a float32
# vector, an int32 vector, and a float32 scalar.
def consume_members(a, b, c):
    return tf.reduce_sum(a) + tf.cast(tf.reduce_sum(b), tf.float32) + c


floats = tf.data.Dataset.from_tensor_slices(tf.ones((8, 4)))
ints = tf.data.Dataset.from_tensor_slices(tf.zeros((8, 3), dtype=tf.int32))
labels = tf.data.Dataset.from_tensor_slices(tf.zeros((8,)))
dataset = tf.data.Dataset.zip((P(floats, ints), labels))

for (a, b), c in dataset:
    consume_members(a, b, c)
