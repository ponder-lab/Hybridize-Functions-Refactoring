import tensorflow as tf


# Receives each element of a dataset built from a generator, a tuple of a float32 and an int32 tensor.
def consume(element):
    return tf.reduce_sum(element[0]) + tf.cast(tf.reduce_sum(element[1]), tf.float32)


def pairs():
    for _ in range(4):
        yield tf.ones((4,)), tf.zeros((3,), dtype=tf.int32)


dataset = tf.data.Dataset.from_generator(
    pairs,
    output_signature=(
        tf.TensorSpec(shape=(4,), dtype=tf.float32),
        tf.TensorSpec(shape=(3,), dtype=tf.int32),
    ),
)

for element in dataset:
    consume(element)
