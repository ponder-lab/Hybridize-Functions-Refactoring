import tensorflow as tf


# Receives each batch of a dataset of pairs, and each element of a dataset mapped from it.
def consume(element):
    return tf.reduce_sum(element[0]) + tf.cast(tf.reduce_sum(element[1]), tf.float32)


dataset = tf.data.Dataset.from_tensor_slices(
    (tf.ones((8, 4)), tf.zeros((8, 3), dtype=tf.int32))
).batch(2)
mapped = dataset.map(lambda x, y: (x, y))

for element in dataset:
    consume(element)

for element in mapped:
    consume(element)
