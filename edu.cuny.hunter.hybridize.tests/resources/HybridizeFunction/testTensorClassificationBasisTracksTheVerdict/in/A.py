# `x` is tensor-typed by the analysis; `factor` is a plain integer at every call site, so the
# analysis runs on both and reaches opposite verdicts. Both verdicts are DETERMINATIONS, which is
# the point: a determined non-tensor must be distinguishable from a parameter whose classification
# never ran, and only the recorded basis can say which happened.
import tensorflow as tf


def scale(x, factor):
    return x * factor


t = tf.ones((2, 2))
scale(t, 3)
