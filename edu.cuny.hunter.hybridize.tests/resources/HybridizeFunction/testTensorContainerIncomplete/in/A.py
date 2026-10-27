# Fixture for a dict that holds a tensor under a key the analysis cannot resolve: `store`
# writes the tensor under a key read from the environment and passes the dict to `read`,
# which receives a dict of tensors at run time.
import os

import tensorflow as tf


def read(d):
    return d


def store(d, k, t):
    d[k] = t
    return read(d)


store({}, os.environ.get("KEY", "a"), tf.ones([2]))
