# Fixture for dicts of tensors keyed by constants that are not strings: a float and a bool.
# Each parameter receives a dict whose only value is a tensor, as does the string-keyed
# control.
import tensorflow as tf


def by_float(d):
    return d[1.5]


def by_bool(d):
    return d[True]


def by_string(d):
    return d["a"]


by_float({1.5: tf.ones([2])})
by_bool({True: tf.ones([2])})
by_string({"a": tf.ones([2])})
