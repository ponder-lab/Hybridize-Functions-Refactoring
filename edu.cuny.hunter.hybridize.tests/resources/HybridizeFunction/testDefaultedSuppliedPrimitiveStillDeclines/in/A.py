# #795 negative: `flag` is defaulted but a call site supplies it with a varying value, so it IS a
# retrace key and the exemption must not apply: `g` stays declined by HAS_PRIMITIVE_PARAMETERS.
import tensorflow as tf


def g(x, flag=1):
    if flag:
        return tf.reduce_sum(x)
    return tf.reduce_sum(x) * 2


g(tf.constant([1.0, 2.0]))
g(tf.constant([3.0, 4.0]), 5)
