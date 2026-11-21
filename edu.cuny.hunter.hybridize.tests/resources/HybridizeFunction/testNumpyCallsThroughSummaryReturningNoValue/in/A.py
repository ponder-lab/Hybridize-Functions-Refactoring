import operator

import numpy as np
import tensorflow as tf

table = np.arange(6)


def keyword(t):
    d = dict(x=t)
    return np.sum(d["x"])


def positional(t):
    d = dict({"x": t})
    return np.sum(d["x"])


def pairs(t):
    d = dict([("x", t)])
    return np.sum(d["x"])


def unpacked(t):
    e = {"x": t}
    d = dict(**e)
    return np.sum(d["x"])


def bound(n):
    return n + np.sum(table[:n])


def stepped(n):
    return n + np.sum(table[::n])


def mixed(n):
    if n > 3:
        n = 2
    return n + np.sum(table[:n])


def indexed(n):
    return n + np.sum(table[: operator.index(n)])


def static_bound(x):
    return tf.reduce_sum(x) + np.sum(table[: x.shape[0]])


def shape_bound(x, n):
    return tf.reduce_sum(x) + np.prod(x.shape[:n])


def shape_bound_swapped(n, x):
    return tf.reduce_sum(x) + np.prod(x.shape[:n])


def either(p, q, first):
    if first:
        return p
    return q


def two_shapes(x, y):
    return tf.reduce_sum(x) + tf.reduce_sum(y) + np.prod(either(x.shape, y.shape, True))


t = tf.ones((3,))
assert keyword(t) == 3
assert positional(t) == 3
assert pairs(t) == 3
assert unpacked(t) == 3
assert bound(tf.constant(3)) == 6
assert stepped(tf.constant(2)) == 8
assert mixed(tf.constant(3)) == 6
assert mixed(tf.constant(4)) == 3
assert indexed(tf.constant(3)) == 6
assert static_bound(t) == 6

m = tf.ones((2, 5))
assert shape_bound(m, tf.constant(1)) == 12
assert shape_bound_swapped(tf.constant(1), m) == 12
assert two_shapes(m, tf.ones((3, 4))) == 32
