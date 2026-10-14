# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1062

import numpy as np
import tensorflow as tf


def wild_arange(x):
    return tf.constant(np.arange(x.shape[0]), tf.float32) + tf.reduce_sum(x)


def wild_linspace(x):
    return tf.constant(np.linspace(0.0, 1.0, x.shape[0]), tf.float32) + tf.reduce_sum(x)


def wild_eye(x):
    return tf.constant(np.eye(x.shape[0]), tf.float32) + tf.reduce_sum(x)


def wild_tile(x):
    return tf.constant(np.tile(np.ones(1), x.shape[0]), tf.float32) + tf.reduce_sum(x)


def wild_random_size(x):
    u = np.random.uniform(size=x.shape[0])
    return tf.constant(u, tf.float32) + tf.reduce_sum(x)


def resolved_axis_arange(x):
    return tf.constant(np.arange(x.shape[1]), tf.float32) + tf.reduce_sum(x)


def ranked_arange(x):
    return tf.constant(np.arange(x.shape[0]), tf.float32) + tf.reduce_sum(x)


for n in (3, 4):
    t = tf.ones((n, 2))
    assert wild_arange(t).shape == (n,)
    assert wild_linspace(t).shape == (n,)
    assert wild_eye(t).shape == (n, n)
    assert wild_tile(t).shape == (n,)
    assert wild_random_size(t).shape == (n,)
    assert resolved_axis_arange(t).shape == (2,)

assert ranked_arange(tf.ones((3, 2))).shape == (3,)
