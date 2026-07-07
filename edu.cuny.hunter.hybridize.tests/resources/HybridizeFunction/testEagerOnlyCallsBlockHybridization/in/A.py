import tensorflow as tf


def fetch(x):
    m = tf.reduce_mean(x)
    return m.numpy()


def outer(x):
    return fetch(x)


def compute(x):
    return tf.reduce_mean(x)


t = tf.constant([1.0, 2.0, 3.0])
assert fetch(t) == 2.0
assert outer(t) == 2.0
assert compute(t).numpy() == 2.0
