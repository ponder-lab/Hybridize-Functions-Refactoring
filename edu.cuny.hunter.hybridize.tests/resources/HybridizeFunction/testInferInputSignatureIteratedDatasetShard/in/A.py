import tensorflow as tf


# Receives each element of a dataset of pairs, sharded two ways: a float32 and an int32 vector.
def consume(element):
    return tf.reduce_sum(element[0]) + tf.cast(tf.reduce_sum(element[1]), tf.float32)


pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones((8, 4)), tf.zeros((8, 3), tf.int32))
)

for element in pairs.shard(2, 0):
    consume(element)
