# Regression fixture for category (b): a parameter classified as tensor-typed by Phase 3
# (`hasTensorContainer`) but with no Phase 2 (Ariadne call-site) shape/dtype evidence in
# `getTensorTypes()`. The function body uses `xs[0]` so Ariadne sees the parameter as a
# list-of-tensors container; `isTensor()` is TRUE, `getTensorTypes()` is empty. Per #508,
# `inferInputSignature` drops the signature and emits a per-parameter INFO referencing #509.
import tensorflow as tf


def f(xs):
    return xs[0]


f([tf.constant([1.0, 2.0])])
