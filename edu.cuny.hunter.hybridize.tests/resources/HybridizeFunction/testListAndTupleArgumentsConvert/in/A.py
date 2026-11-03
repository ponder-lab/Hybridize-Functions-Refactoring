# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1184
# Controls for the set refusal: `tf.function` keys a list or a tuple argument by its contents, so a list or a tuple of tensors crosses
# the boundary, and each function converts.
# - `listed`: passed a list of tensors.
# - `tupled`: passed a tuple of tensors.
import tensorflow as tf


def listed(x, s):
    return x * len(s)


def tupled(x, s):
    return x * len(s)


t = tf.ones([2])
listed(t, [t, tf.zeros([2])])
tupled(t, (t, tf.zeros([2])))
