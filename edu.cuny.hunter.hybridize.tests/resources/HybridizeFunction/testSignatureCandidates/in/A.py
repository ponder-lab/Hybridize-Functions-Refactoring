import tensorflow as tf


@tf.function
def a(x):
    return x * 2


def b(x):
    return tf.reduce_sum(x)


def d(x):
    return tf.reduce_mean(x)


def c(x):
    return x.numpy()


a(tf.ones(3))
b(tf.ones(3))
b(tf.ones(3, dtype=tf.int32))
c(tf.ones(3))
d(tf.ones(3))
