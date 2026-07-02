import tensorflow as tf


@tf.function
def barren(x):
    return x


@tf.function
def compute(x):
    return tf.reduce_sum(x)


t = tf.constant([1.0, 2.0, 3.0])
barren(t)
compute(t)
