# Fixture for the sequence reduction (#781), previously the category (b) container-drop pin.
# The function body uses `xs[0]` so Ariadne sees the parameter as a list-of-tensors container;
# `isTensor()` is TRUE via Phase 3 while `getTensorTypes()` stays empty (the parameter is a
# list, not a tensor). The element's concrete type ((2,) float32 from the call site) is
# surfaced per position and reduces to the nested signature
# `[[tf.TensorSpec(shape=(2,), dtype=tf.float32)]]`.
import tensorflow as tf


def f(xs):
    return xs[0]


f([tf.constant([1.0, 2.0])])
