# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1184
# A call passing a set to a `tf.function` raises `TypeError: Python object could not be represented through the generic tracing
# type`, whatever the set holds.
# - `scaled`: already hybrid and passed a set of floats, so the call raises as written.
# - `listed`: already hybrid and passed a list of tensors, the control.
import tensorflow as tf


@tf.function
def scaled(x, s):
    return x * len(s)


@tf.function
def listed(x, l):
    return x * len(l)


try:
    scaled(tf.ones([2]), {1.0, 2.0})
except TypeError:
    pass

listed(tf.ones([2]), [tf.ones([2]), tf.zeros([2])])
