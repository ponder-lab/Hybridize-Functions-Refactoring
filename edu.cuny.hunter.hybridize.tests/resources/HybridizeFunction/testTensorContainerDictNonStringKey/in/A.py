# Fixture for a dict of tensors keyed by a constant that is neither a string nor an integer.
# Each parameter receives a dict whose only value is a tensor.
import tensorflow as tf


def by_float(d):
    return d[1.5]


def by_string(d):
    return d["a"]


by_float({1.5: tf.ones([2])})
by_string({"a": tf.ones([2])})
