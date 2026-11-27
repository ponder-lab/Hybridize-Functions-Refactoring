# Fixture for #1323: every call passing `scale` passes its default, so the inferred signature leaves
# `scale` out. But the calls disagree on the last extent, which `x.shape[-1]` reads statically to
# decide a branch, so the signature is withheld (#864), and no argument is deleted or reported deleted.
import tensorflow as tf


def f(x, scale=2.0):
    if x.shape[-1] != 4:
        return x * 3.0

    return x * scale


f(tf.ones([2, 4]), scale=2.0)
f(tf.ones([2, 5]))
