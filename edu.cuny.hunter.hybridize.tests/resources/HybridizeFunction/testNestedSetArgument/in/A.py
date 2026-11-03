# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1184
# `tf.function` flattens a list or a dictionary argument and keys each element, so a set inside one makes the call raise a `TypeError`
# just as a set passed directly does.
# - `listed`: passed a list holding a set of tensor references.
# - `keyed`: passed a dictionary holding a set of tensor references.
import tensorflow as tf


def listed(x, c):
    return x * len(c[0])


def keyed(x, c):
    return x * len(c["s"])


t = tf.ones([2])
listed(t, [{t.ref()}])
keyed(t, {"s": {t.ref()}})
