# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1184
# `tf.function` can't build a trace key for a set, so a call passing one raises `TypeError: Python object could not be represented
# through the generic tracing type`, whatever the set holds.
# - `scaled`: passed a set of floats positionally.
# - `scaled_by_keyword`: passed a set of floats by keyword.
import tensorflow as tf


def scaled(x, s):
    return x * len(s)


def scaled_by_keyword(x, s):
    return x * len(s)


scaled(tf.ones([2]), {1.0, 2.0})
scaled_by_keyword(tf.ones([2]), s={1.0, 2.0})
