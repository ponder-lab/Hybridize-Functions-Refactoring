# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1184
# A tensor is unhashable, so a program keeps tensors in a set by their references. `tf.function` can't build a trace key for the set,
# so a call passing one raises a `TypeError`, although no element is a Python primitive.
# - `scaled`: passed a set of tensor references.
# - `boxed`: passed a set of objects of a program class.
import tensorflow as tf


class Box:
    def __init__(self, n):
        self.n = n


def scaled(x, s):
    return x * len(s)


def boxed(x, s):
    return x * len(s)


t = tf.ones([2])
scaled(t, {t.ref(), tf.zeros([2]).ref()})
boxed(t, {Box(1), Box(2)})
