import tensorflow as tf


def add(a, b):
    return a + b


c = add(tf.Variable([1.0, 2.0]), tf.Variable([2.0, 2.0]))
