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
