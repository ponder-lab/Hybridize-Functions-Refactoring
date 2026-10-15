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
    return x * float(tf.reduce_sum(x))


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


def local_in_branch(x):
    if tf.reduce_sum(x) > 0.0:
        tmp = x + 1.0
        x = tmp * 2.0
    return x


def primitive_branch(x, n):
    if n > 0:
        return None
    return x + float(n)


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
local_in_branch(t)
primitive_branch(t, 0)
