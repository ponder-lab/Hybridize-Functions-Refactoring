# The function `func` closes over `bias`, a module-level tensor with a *distinct* shape from `t`.
# The single parameter `t` is the only tensor that should appear in the inferred signature; `bias`
# is not a parameter and must not leak in. Shape distinction makes the leak observable: an
# implementation that collected all tensors and deduplicated by `TensorType` would surface both
# shapes, failing the singleton assertion.

import tensorflow as tf

bias = tf.constant(
    [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
)  # shape (3, 2), distinct from `t`'s (2,)


def func(t):
    return t + tf.reduce_sum(bias)


x = tf.constant([1.0, 2.0])
assert x.dtype == tf.float32
assert x.shape == (2,)
assert bias.shape == (3, 2)
func(x)
