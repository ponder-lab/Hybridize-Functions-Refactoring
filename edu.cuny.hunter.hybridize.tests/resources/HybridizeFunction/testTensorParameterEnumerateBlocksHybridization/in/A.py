# Fixture for #1098 (distilling TensorFlow-Examples' gradient_boosted_trees.py
# `to_binary_class`): iterating a tensor parameter through a builtin that iterates
# its argument (`enumerate`, `zip`) works eagerly but raises
# OperatorNotAllowedInGraphError under tracing, so the conversion must be
# declined. Enumerating a Python list of tensors remains convertible.
import copy

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


def enumerate_list(a, b):
    total = tf.zeros(())
    for i, t in enumerate([a, b]):
        total = total + tf.reduce_sum(t)
    return total


y_train_binary = to_binary_class(copy.deepcopy(y_train))
assert set(y_train_binary) <= {0.0, 1.0}

t = tf.constant([1.0, 2.0, 3.0])
assert float(enumerate_param(t)) == 6.0
assert float(zip_param(t, t)) == 14.0
assert float(reversed_param(t)) == 6.0
assert float(enumerate_list(t, t)) == 12.0
