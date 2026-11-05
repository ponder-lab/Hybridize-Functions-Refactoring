import tensorflow as tf


# Receives each element of a dataset flat-mapped from a dataset of pairs, a float32 vector of four and an int32 vector of three.
def consume(element):
    return tf.reduce_sum(element[0]) + tf.cast(tf.reduce_sum(element[1]), tf.float32)


pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones((8, 4)), tf.zeros((8, 3), tf.int32))
)
dataset = pairs.flat_map(lambda a, b: tf.data.Dataset.from_tensors((a, b)))

for element in dataset:
    consume(element)
