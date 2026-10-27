# A broader supplied signature on a function no reachable code calls (#808). Nothing is inferred without a call, so there is nothing to
# narrow to, and the signature is left unchanged.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return t + 1
