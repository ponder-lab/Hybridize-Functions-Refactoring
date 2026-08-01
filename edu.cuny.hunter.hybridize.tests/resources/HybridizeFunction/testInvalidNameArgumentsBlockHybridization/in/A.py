import tensorflow as tf


def evaluate(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 1, keepdims=True), tf.float32)


def annotate(x):
    return tf.sqrt(x, name=tf.float32)


def compute(x):
    return tf.sqrt(x, name="root")


t = tf.ones((2, 3))
assert evaluate(t).shape == (2, 1)
assert annotate(t).shape == (2, 3)
assert compute(t).shape == (2, 3)
