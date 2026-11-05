import tensorflow as tf
import tensorflow_datasets as tfds


# Receives each element of a supervised dataset zipped with another dataset, an ((image, label), weight) pair.
def consume(element):
    (image, label), weight = element
    return tf.reduce_sum(tf.cast(image, tf.float32)) + tf.cast(
        label, tf.float32
    ) * tf.reduce_sum(weight)


# Receives the members of each element of a supervised dataset zipped with another dataset.
def consume_members(image, label, weight):
    return tf.reduce_sum(tf.cast(image, tf.float32)) + tf.cast(
        label, tf.float32
    ) * tf.reduce_sum(weight)


supervised = tfds.load("mnist", split="train", as_supervised=True)
weights = tf.data.Dataset.from_tensor_slices(tf.ones((8, 4)))
dataset = tf.data.Dataset.zip((supervised, weights))

for element in dataset:
    consume(element)

for (image, label), weight in dataset:
    consume_members(image, label, weight)
