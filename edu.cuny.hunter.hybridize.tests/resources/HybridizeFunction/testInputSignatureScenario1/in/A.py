# Smallest input-signature inference case: a function called once with a single tensor of
# concrete dtype and shape. The inferred signature is exactly that single (dtype, shape) tuple,
# wrapped as a singleton InputSignature.

import tensorflow as tf


def func(t):
    return t + 1


func(tf.constant([1.0, 2.0]))
