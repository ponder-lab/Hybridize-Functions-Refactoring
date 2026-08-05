import tensorflow as tf


def evaluate(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x), 1, keepdims=True), tf.float32)


def annotate(x):
    return tf.sqrt(x, name=tf.float32)


def scale(x):
    return tf.sqrt(x, 2)


def toggle(x):
    return tf.sqrt(x, name=True)


def compute(x):
    return tf.sqrt(x, name="root")


t = tf.ones((2, 3))
assert evaluate(t).shape == (2, 1)
assert annotate(t).shape == (2, 3)
assert scale(t).shape == (2, 3)
assert toggle(t).shape == (2, 3)
assert compute(t).shape == (2, 3)
