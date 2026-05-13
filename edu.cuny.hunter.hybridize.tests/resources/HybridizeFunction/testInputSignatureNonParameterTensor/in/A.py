# The function `func` closes over `bias`, a module-level tensor. The single parameter `t` is also
# a tensor (provided at the call site). The inferred input signature should contain only `t`'s
# tensor type; `bias` is not a parameter and must not leak into the signature.

import tensorflow as tf

bias = tf.constant([10.0, 20.0])


def func(t):
    return t + bias


func(tf.constant([1.0, 2.0]))
