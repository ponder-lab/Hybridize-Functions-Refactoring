import tensorflow as tf


def compute(x):
    return tf.reduce_sum(x)


first = compute(tf.ones((2, 2)))


def compute(x):
    return tf.reduce_mean(x)


second = compute(tf.ones((3, 3)))
