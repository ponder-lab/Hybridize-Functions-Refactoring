# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1061

import tensorflow as tf


def wild_lambda(x):
    f = lambda: tf.ones((x.shape[0], 1))
    return f() + tf.reduce_sum(x)


def wild_nested_def(x):
    def g():
        return tf.reshape(x, (x.shape[0], 1))

    return g()


def wild_comprehension(x):
    stacked = tf.stack([tf.zeros((x.shape[0],)) + k for k in (1.0, 2.0)])
    return stacked + tf.reduce_sum(x)


def ranked_nested_def(x):
    def g():
        return tf.reshape(x, (x.shape[0], 1))

    return g()


for n in (2, 3):
    t = tf.ones((n,))
    assert wild_lambda(t).shape == (n, 1)
    assert wild_nested_def(t).shape == (n, 1)
    assert wild_comprehension(t).shape == (2, n)

assert ranked_nested_def(tf.ones((2,))).shape == (2, 1)
