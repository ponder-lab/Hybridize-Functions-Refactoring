import tensorflow as tf


@tf.function
def testing_step(a):
    return tf.reduce_sum(tf.constant([1.0, 2.0]))


testing_step(True)
