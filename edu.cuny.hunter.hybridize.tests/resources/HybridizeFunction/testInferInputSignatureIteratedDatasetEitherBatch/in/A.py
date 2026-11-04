import sys

import tensorflow as tf


# Receives each batch of a dataset that is either of two: batches of float32 vectors or batches of int32 vectors.
def consume(x):
    return tf.reduce_sum(tf.cast(x, tf.float32))


floats = tf.data.Dataset.from_tensor_slices(tf.ones((8, 4)))
ints = tf.data.Dataset.from_tensor_slices(tf.zeros((8, 3), dtype=tf.int32))
flag = len(sys.argv) > 1
dataset = (floats if flag else ints).batch(2)

for x in dataset:
    consume(x)
