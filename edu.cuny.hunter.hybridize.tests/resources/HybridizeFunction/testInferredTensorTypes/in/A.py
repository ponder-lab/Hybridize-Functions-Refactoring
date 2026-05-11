# Ported from wala/ML's `tf2_test_function8.py`. Ariadne's `testFunction8`
# (`com.ibm.wala.cast.python.ml.test.TestTensorflow2Model`) asserts that the
# parameter `t` of `func` is inferred to have two TensorTypes -- same dtype
# (float32), divergent shape ((2, 1) and (2,)). Used here to exercise
# `Parameter.getTensorTypes()` for the shape-divergence/same-dtype scenario.
import tensorflow as tf
from random import random


def func(t):
    pass


n = random()

if n > 0.5:
    l = [[1.0], [3.0]]
else:
    l = [1.0, 3.0]

a = tf.constant(l)
assert a.dtype == tf.float32
assert a.shape == (2, 1) or a.shape == (2,)

func(a)
