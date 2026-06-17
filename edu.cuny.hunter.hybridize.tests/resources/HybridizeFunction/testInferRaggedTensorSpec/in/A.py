import tensorflow as tf


@tf.function
def f(t):
    return t


f(tf.ragged.constant([[1, 2, 3], [4, 5]]))
