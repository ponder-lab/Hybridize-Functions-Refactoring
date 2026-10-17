# Fixture for #1098 (distilling TensorFlow-Examples' gradient_boosted_trees.py
# `to_binary_class`): iterating a tensor parameter through a wrapper that iterates
# its argument (`enumerate`, `zip`), and storing into a tensor parameter by
# subscript, both work eagerly on a NumPy array but raise under tracing
# (OperatorNotAllowedInGraphError; TypeError: 'Tensor' object does not support
# item assignment), so the conversion must be declined. Enumerating a Python list
# of tensors, storing into a list built in the body, and setting an attribute on a
# tensor, which tracing supports, remain convertible.
import copy

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


def to_binary_class(y):
    for i, label in enumerate(y):
        if label >= 23.0:
            y[i] = 1
        else:
            y[i] = 0
    return y


def enumerate_param(x):
    total = tf.zeros(())
    for i, e in enumerate(x):
        total = total + e
    return total


def zip_param(x, y):
    total = tf.zeros(())
    for a, b in zip(x, y):
        total = total + a * b
    return total


def reversed_param(x):
    total = tf.zeros(())
    for e in reversed(x):
        total = total + e
    return total


def set_first(y):
    y[0] = 1
    return tf.reduce_sum(y)


def add_to_first(y):
    y[0] += 1.0
    return tf.reduce_sum(y)


def store_into_derived(y):
    z = y * 2.0
    z[0] = 0.0
    return tf.reduce_sum(z)


def enumerate_list(a, b):
    total = tf.zeros(())
    for i, t in enumerate([a, b]):
        total = total + tf.reduce_sum(t)
    return total


def store_into_local_list(x):
    out = [x, x]
    out[0] = x * 2.0
    return out[0] + out[1]


def tag_derived(x):
    y = x * 2.0
    y.tag = 1
    return y


def store_string_key(y):
    z = y * 2.0
    z["k"] = 0.0
    return z


def store_multi_index(y):
    z = y * 2.0
    z[0, 1] = 0.0
    return tf.reduce_sum(z)


y_train_binary = to_binary_class(copy.deepcopy(y_train))
assert set(y_train_binary) <= {0.0, 1.0}

t = tf.constant([1.0, 2.0, 3.0])
assert float(enumerate_param(t)) == 6.0
assert float(zip_param(t, t)) == 14.0
assert float(reversed_param(t)) == 6.0
assert float(set_first(np.array([10.0, 30.0]))) == 31.0
assert float(add_to_first(np.array([10.0, 30.0]))) == 41.0
assert float(store_into_derived(np.array([10.0, 30.0]))) == 60.0
assert float(enumerate_list(t, t)) == 12.0
assert float(tf.reduce_sum(store_into_local_list(t))) == 18.0
assert float(tf.reduce_sum(tag_derived(t))) == 12.0
assert float(store_multi_index(np.array([[1.0, 2.0], [3.0, 4.0]]))) == 16.0

try:
    store_string_key(np.array([10.0, 30.0]))
    raise AssertionError("A NumPy array has no string-keyed items.")
except IndexError:
    pass
