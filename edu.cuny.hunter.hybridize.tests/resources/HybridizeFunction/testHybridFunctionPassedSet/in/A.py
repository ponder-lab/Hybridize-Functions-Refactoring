# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1184
# A call passing a set to a `tf.function` raises `TypeError: Python object could not be represented through the generic tracing
# type`, whatever the set holds. Each function below is already hybrid.
# - `scaled`: passed a tensor and a set of floats, and computes with the tensor.
# - `counted`: passed only a set of floats, so it has no tensor parameter.
# - `barren`: passed a tensor and a set of floats, but performs no tensor computation.
# - `listed`: passed a tensor and a list of tensors, the control.
import tensorflow as tf


@tf.function
def scaled(x, s):
    return x * len(s)


@tf.function
def counted(s):
    return len(s)


@tf.function
def barren(x, s):
    return len(s)


@tf.function
def listed(x, l):
    return x * len(l)


try:
    scaled(tf.ones([2]), {1.0, 2.0})
except TypeError:
    pass

try:
    counted({1.0, 2.0})
except TypeError:
    pass

try:
    barren(tf.ones([2]), {1.0, 2.0})
except TypeError:
    pass

listed(tf.ones([2]), [tf.ones([2]), tf.zeros([2])])
