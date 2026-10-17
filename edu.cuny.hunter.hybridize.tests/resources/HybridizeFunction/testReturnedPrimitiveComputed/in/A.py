# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1117

import tensorflow as tf


def scaled(x):
    n = 3
    return x * 2.0, n * 2


def counted(x, xs):
    return x * 2.0, len(xs)


def doubled_count(x, xs):
    m = len(xs)
    return x * 2.0, m * 2


t = tf.constant([1.0, 2.0])

y, k = scaled(t)
print({k: "scaled"}[6])

y, k = counted(t, [t, t])
print({k: "counted"}[2])

y, k = doubled_count(t, [t, t])
print({k: "doubled_count"}[4])
