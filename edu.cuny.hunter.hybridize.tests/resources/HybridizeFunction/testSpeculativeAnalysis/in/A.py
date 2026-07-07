import tensorflow as tf


@tf.function
def training_step(a):
    return tf.reduce_sum(tf.constant([1.0, 2.0]))


training_step(True)
