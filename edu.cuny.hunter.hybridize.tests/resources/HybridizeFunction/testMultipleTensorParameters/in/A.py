# Regression fixture for #498. The function `f` has two tensor parameters (`a`, `b`).
# After `Parameter.classifyAsTensor` runs, both have `isTensor() == TRUE`, and
# `Function.getHasTensorParameter() == TRUE`. Pins that ALL non-self tensor parameters
# classify independently—the classification is per-parameter, not first-match-wins.

import tensorflow as tf


def f(a, b):
    return a + b


x = tf.constant([1.0, 2.0])
y = tf.constant([3.0, 4.0])
assert x.shape == (2,)
assert y.shape == (2,)
f(x, y)
