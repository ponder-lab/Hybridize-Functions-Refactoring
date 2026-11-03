# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1184
# A call passing a set to a `tf.function` raises `TypeError: Python object could not be represented through the generic tracing
# type`, whatever the set holds. Here the set is built by calling `set`.
# - `scaled`: passed `set([1.0, 2.0])`.
import tensorflow as tf


def scaled(x, s):
    return x * len(s)


scaled(tf.ones([2]), set([1.0, 2.0]))
