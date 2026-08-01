import tensorflow as tf


def inner(x):
    return tf.reduce_sum(x)


@tf.function
def outer(x):
    return inner(x) * 2.0


def mixed(x):
    return tf.reduce_sum(x)


@tf.function
def outer2(x):
    return mixed(x)


t = tf.ones((2, 2))
assert outer(t).numpy() == 8.0
assert outer2(t).numpy() == 4.0
assert mixed(t).numpy() == 4.0
