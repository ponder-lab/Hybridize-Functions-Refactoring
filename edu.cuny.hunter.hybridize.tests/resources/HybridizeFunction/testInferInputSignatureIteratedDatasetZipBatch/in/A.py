import tensorflow as tf


# Receives the first member of each batch of two zipped datasets of eight rows each, so every batch is full.
def consume(a):
    return tf.reduce_sum(a)


floats = tf.data.Dataset.from_tensor_slices(tf.ones((8, 4)))
ints = tf.data.Dataset.from_tensor_slices(tf.zeros((8, 3), dtype=tf.int32))
dataset = tf.data.Dataset.zip((floats, ints)).batch(2)

for a, b in dataset:
    consume(a)
