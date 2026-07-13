import os

import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def head_over_unresolved(x, w):
    dims = get_shape(x)
    inner = np.prod(dims[:1])
    flat = tf.reshape(x, [-1, inner])
    return tf.matmul(flat, w)


# `x`'s leading axis is an environment-sourced fixed size the analysis cannot
# compute, so it types `UnresolvedDim` (a fixed run-time size with no
# run-time-`None` evidence), not `DynamicDim`. numpy over it is permitted under
# permit-⊤, the DenseLayer config-dimension case option D recovers (#772,
# wala/ML#721). Contrast the `DynamicDim` (`tf.keras.Input`) feeds in
# testNumpyOnDynamicShape*, which decline.
n = int(os.environ.get("ARIADNE_TEST_N", "8"))
x = tf.ones((n, 5))
w = tf.ones((5, 2))
head_over_unresolved(x, w)
