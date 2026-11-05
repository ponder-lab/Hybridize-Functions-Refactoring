# Fixture for a dict that holds tensors under computed keys: `fill` stores a tensor under
# `str(i)` for each `i`, a key the analysis can't list, and passes the dict to `read`, which
# receives a dict of tensors at run time.
import tensorflow as tf


def read(d):
    return d


def fill(n):
    d = {}
    for i in range(n):
        d[str(i)] = tf.ones([2])
    return read(d)


fill(3)
