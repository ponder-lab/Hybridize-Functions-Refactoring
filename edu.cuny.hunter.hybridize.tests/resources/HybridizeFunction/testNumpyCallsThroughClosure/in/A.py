# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1061

import numpy as np
import tensorflow as tf


def helper(y):
    return np.sum(y)


def via_comprehension(x):
    return np.array([np.sum(x) + k for k in (1.0, 2.0)])


def via_lambda(x):
    f = lambda: np.sum(x)
    return f()


def via_nested_def(x):
    def g():
        return np.sum(x)

    return g()


def via_nested_callee(x):
    def g():
        return helper(x)

    return g()


def via_nested_def_arguments(mu, phi):
    def f(m, p):
        return np.log(p) + np.sum(m)

    return np.argmax(np.array([f(mu[i], phi[i]) for i in range(2)]))


def via_own_exposed_read(x):
    g = lambda: tf.reduce_sum(x)
    return np.sum(x) + g()


def clean_closure(x):
    def g():
        return tf.reduce_sum(x)

    return g()


def closure_over_local(x):
    c = np.ones(3)

    def g():
        return np.sum(c)

    return g() + tf.reduce_sum(x)


t = tf.constant([1.0, 2.0, 3.0])
assert via_comprehension(t).shape == (2,)
assert float(via_lambda(t)) == 6.0
assert float(via_nested_def(t)) == 6.0
assert float(via_nested_callee(t)) == 6.0
m = tf.constant([[1.0], [2.0]])
assert via_nested_def_arguments(m, tf.constant([1.0, 2.0])) == 1
assert float(via_own_exposed_read(t)) == 12.0
assert float(clean_closure(t)) == 6.0
assert float(closure_over_local(t)) == 9.0
