# Fixture for #1098 (distilling TensorFlow-Examples' gradient_boosted_trees.py
# `to_binary_class`): passing a tensor parameter to a builtin that iterates it
# (`enumerate`, `zip`, `list`, ...) works eagerly but raises under tracing, so the
# conversion must be declined. `sorted` runs under tracing, since AutoGraph
# converts it, and enumerating a Python list of tensors remains convertible.
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


def list_param(x):
    return tf.stack(list(x))


def tuple_param(x):
    return tf.stack(tuple(x))


def iter_param(x):
    return next(iter(x)) * 2.0


def sum_param(x):
    return sum(x) * 2.0


def min_param(x):
    return min(x) * 2.0


def max_param(x):
    return max(x) * 2.0


def any_param(x):
    return tf.constant(any(x))


def all_param(x):
    return tf.constant(all(x))


def sorted_param(x):
    return tf.stack(sorted(x))


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
assert list(list_param(t).numpy()) == [1.0, 2.0, 3.0]
assert list(tuple_param(t).numpy()) == [1.0, 2.0, 3.0]
assert float(iter_param(t)) == 2.0
assert float(sum_param(t)) == 12.0
assert float(min_param(t)) == 2.0
assert float(max_param(t)) == 6.0
assert bool(any_param(t))
assert bool(all_param(t))
assert list(sorted_param(tf.constant([3.0, 1.0, 2.0])).numpy()) == [1.0, 2.0, 3.0]
