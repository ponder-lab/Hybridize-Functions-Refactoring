import tensorflow as tf
import tensorflow_datasets as tfds


# Receives each element of a supervised dataset, an (image, label) pair.
def consume(element):
    return tf.reduce_sum(tf.cast(element[0], tf.float32)) + tf.cast(
        element[1], tf.float32
    )


# Receives the members of each element of a supervised dataset.
def consume_members(image, label):
    return tf.reduce_sum(tf.cast(image, tf.float32)) + tf.cast(label, tf.float32)


dataset = tfds.load("mnist", split="train", as_supervised=True)

for element in dataset:
    consume(element)

for image, label in dataset:
    consume_members(image, label)
