# Fixture for #787's suffix rule. `training` is non-tensor, defaulted, and supplied by nobody, so
# it looks omittable, but `mask` follows it and IS a tensor needing a spec. `input_signature`
# covers parameters by position, so emitting `[TensorSpec(x), TensorSpec(mask)]` would apply
# mask's spec to `training`. The parameter must therefore block, as
# DEFAULTED_PARAMETER_PRECEDES_SPEC.
#
# `mask` is passed by keyword deliberately: passing it positionally would have to pass `training`
# too, which would make this the DEFAULTED_PARAMETER_SUPPLIED case instead and never reach the
# suffix rule.
#
# Python's grammar forces defaults to trail the required parameters, but it permits a defaulted
# tensor parameter after a defaulted non-tensor one, which is this shape. It is the ordinary Keras
# `call(self, x, training=False, mask=None)` signature, so the rule is not hypothetical.
import tensorflow as tf


def f(x, training=False, mask=None):
    if training:
        return tf.reduce_sum(x)
    return tf.reduce_sum(x * mask)


f(tf.constant([1.0, 2.0]), mask=tf.constant([1.0, 0.0]))
