# Regression fixture for #495. With `experimental_follow_type_hints=True` and a tensor-typed
# type hint, `Parameter.isTensorTyped`'s Phase 1 returns true before Phase 2 (the Ariadne
# query) runs. Before #496, that early return left the per-Parameter tensor-types cache
# unpopulated, so downstream consumers reading `Parameter.getTensorTypes()` (no-arg) saw an
# empty set even when Ariadne had a concrete `TensorType` for the parameter from the call
# site. Hoisting `inferTensorTypes` to fire before Phase 1 closes the hole.

import tensorflow as tf


@tf.function(experimental_follow_type_hints=True)
def func(t: tf.Tensor):
    return t + 1


x = tf.constant([1.0, 2.0])
assert x.dtype == tf.float32
assert x.shape == (2,)
func(x)
