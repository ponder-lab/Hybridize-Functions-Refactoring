# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1074

import tensorflow as tf


def none_return(x):
    if tf.reduce_sum(x) > 0.0:
        return None
    return x + 1.0


def none_merge(x):
    y = None
    if tf.reduce_sum(x) > 0.0:
        y = x + 1.0
    return y


def float_of(x):
    return tf.constant(float(tf.reduce_sum(x)))


def bool_of(x):
    if bool(tf.reduce_sum(x) > 0.0):
        return x + 1.0
    return x - 1.0


def via_callee(x):
    r = none_return(x)
    return x if r is None else r


def tensor_branches(x):
    if tf.reduce_sum(x) > 0.0:
        return x + 1.0
    return x - 1.0


def int_of(x):
    return tf.zeros((2,)) + tf.cast(int(tf.reduce_sum(x)), tf.float32)


def static_none(x):
    if x.shape[0] == 3:
        return x + 1.0
    return None


t = tf.constant([1.0, -2.0])
for f in (
    none_return,
    none_merge,
    float_of,
    bool_of,
    via_callee,
    tensor_branches,
    int_of,
):
    f(t)
static_none(tf.constant([1.0, 2.0, 3.0]))
