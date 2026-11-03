import tensorflow as tf


# Receives each batch of a dataset of pairs, a tuple of a float32 and an int32 tensor.
def consume(element):
    return tf.reduce_sum(element[0]) + tf.cast(tf.reduce_sum(element[1]), tf.float32)


dataset = tf.data.Dataset.from_tensor_slices(
    (tf.ones((8, 4)), tf.zeros((8, 3), dtype=tf.int32))
).batch(2)

for element in dataset:
    consume(element)
