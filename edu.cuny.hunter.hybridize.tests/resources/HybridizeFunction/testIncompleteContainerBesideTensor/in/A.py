# Fixture for a parameter that receives a tensor at one call and a dict holding tensors under
# computed keys at another: `fill` stores a tensor under `str(i)` for each `i`, a key the
# analysis can't list, and passes the dict to `read`, which is also passed a tensor directly.
import tensorflow as tf


def read(d):
    return d


def fill(n):
    d = {}
    for i in range(n):
        d[str(i)] = tf.ones([2])
    return read(d)


fill(3)
read(tf.ones([2]))
