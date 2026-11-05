import tensorflow as tf


# Receives the pair in each element of an enumerated dataset of pairs: a float32 and an int32 tensor.
def consume(element):
    return tf.reduce_sum(element[0]) + tf.cast(tf.reduce_sum(element[1]), tf.float32)


# Receives the index in each element of that dataset, an int64 scalar.
def consume_index(index):
    return tf.cast(index, tf.float32)


dataset = tf.data.Dataset.from_tensor_slices(
    (tf.ones((8, 4)), tf.zeros((8, 3), dtype=tf.int32))
).batch(2)

for index, element in dataset.enumerate():
    consume(element)
    consume_index(index)
