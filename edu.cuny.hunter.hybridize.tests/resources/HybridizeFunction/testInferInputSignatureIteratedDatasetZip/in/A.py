import tensorflow as tf


# Receives each element of two batched datasets zipped together, a tuple of a float32 and an int32 tensor.
def consume(element):
    return tf.reduce_sum(element[0]) + tf.cast(tf.reduce_sum(element[1]), tf.float32)


features = tf.data.Dataset.from_tensor_slices(tf.ones((8, 4))).batch(2)
labels = tf.data.Dataset.from_tensor_slices(tf.zeros((8, 3), dtype=tf.int32)).batch(2)
dataset = tf.data.Dataset.zip((features, labels))

for element in dataset:
    consume(element)
