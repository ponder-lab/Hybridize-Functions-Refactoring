import tensorflow as tf


@tf.function
def covered_h(x):
    return tf.reduce_sum(x)


@tf.function
def outer_h(x):
    return covered_h(x) * 2.0


t = tf.ones((2, 2))
assert outer_h(t).numpy() == 8.0
