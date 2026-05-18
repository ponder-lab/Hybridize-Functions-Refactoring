# Regression fixture for #498. The function `f` has two parameters: `t` is a tensor (passed
# from the call site as `tf.constant([1.0, 2.0])`) and `n` is a non-tensor (an int literal).
# After `Parameter.classifyAsTensor` runs (transitively via `Function.inferTensorParameters`),
# `t.isTensor()` returns TRUE and `n.isTensor()` returns FALSE. Pins the classifier→query
# contract: the cached `isTensor()` value reflects the classifier's verdict.

import tensorflow as tf


def f(t, n):
    return t + n


x = tf.constant([1.0, 2.0])
assert x.dtype == tf.float32
assert x.shape == (2,)
f(x, 5)
