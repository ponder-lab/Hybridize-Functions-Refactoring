# Fixture for #501. Parameter `a` is reached via a list-of-tensors call site, so Phase 3
# container detection classifies it as tensor-typed. This pins the cross-method invariant
# that `isTensorContainer() == TRUE` implies `isTensor() == TRUE`, while `getTensorTypes()`
# stays empty (Ariadne emits no single TensorType for the container itself).

import tensorflow as tf


def f(a):
    pass


f([tf.constant([1.0, 2.0])])
