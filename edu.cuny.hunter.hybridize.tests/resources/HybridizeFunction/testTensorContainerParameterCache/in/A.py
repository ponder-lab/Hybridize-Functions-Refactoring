# Regression fixture for #497. Parameter `a` is reached via a list-of-tensors call site;
# Ariadne's per-parameter iterator does not emit a single TensorType for the container
# itself, but `Parameter.isTensorTyped`'s Phase 3 (`hasTensorContainer`) classifies it as
# tensor-typed. The new `isTensorContainer()` cache exposes Phase 3's verdict; the existing
# `getTensorTypes()` cache stays empty (the asymmetry this PR documents).

import tensorflow as tf


def f(a):
    pass


f([tf.constant([1.0, 2.0])])
