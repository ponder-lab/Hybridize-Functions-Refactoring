# Regression fixture for #508 category (a). The function `f` has a tensor parameter `t` (Phase-2 hit
# via the `tf.constant(...)` call site) and a non-tensor parameter `n` (Python `int` literal at the
# call site). `n.isTensor()` is FALSE, blocking input-signature inference: per #508, the function
# emits a per-parameter INFO refactoring status with the source-side recovery suggestion and
# `inferInputSignature()` returns `Optional.empty`.
import tensorflow as tf


def f(t, n):
    return t


f(tf.constant([1.0, 2.0]), 5)
