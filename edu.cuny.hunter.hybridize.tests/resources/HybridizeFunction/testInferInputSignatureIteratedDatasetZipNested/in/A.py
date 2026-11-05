import tensorflow as tf


# Receives each element of a zip whose first member is itself a pair of datasets, unpacked as `(x, y), z`.
def consume(element):
    (x, y), z = element
    return tf.reduce_sum(x) + tf.cast(tf.reduce_sum(y), tf.float32) + z


# Receives the members of that element: a float32 vector, an int32 vector, and a float32 scalar.
def consume_members(x, y, z):
    return tf.reduce_sum(x) + tf.cast(tf.reduce_sum(y), tf.float32) + z


floats = tf.data.Dataset.from_tensor_slices(tf.ones((8, 4)))
ints = tf.data.Dataset.from_tensor_slices(tf.zeros((8, 3), dtype=tf.int32))
labels = tf.data.Dataset.from_tensor_slices(tf.zeros((8,)))
dataset = tf.data.Dataset.zip(((floats, ints), labels))

for element in dataset:
    consume(element)
    (x, y), z = element
    consume_members(x, y, z)
