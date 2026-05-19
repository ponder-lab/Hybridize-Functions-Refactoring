# Regression fixture for category (b): a parameter classified as tensor-typed by Phase 1
# (`hasTensorTypeHint`) but with no Phase 2 (Ariadne call-site) shape/dtype evidence in
# `getTensorTypes()`. The parameter `x` has a `tf.Tensor` type hint, classifying it as
# tensor-typed; the call site supplies a non-tensor (`int`), so Ariadne's per-parameter
# cache stays empty. Per #508, `inferInputSignature` drops the signature and emits a
# per-parameter INFO referencing #509.
import tensorflow as tf


def f(x: tf.Tensor):
    pass


f(5)
