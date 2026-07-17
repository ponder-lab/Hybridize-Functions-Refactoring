# #795: `flag` is a defaulted primitive that no call site supplies, so it is always the constant
# default (1), not a varying trace key, and must not decline `f` for HAS_PRIMITIVE_PARAMETERS.
# Module-level, so the default is materialized at 0.52.34 (unlike a trampolined method), which is
# why this exercises the exemption directly: without it, `f` is declined; with it, `f` is optimizable.
import tensorflow as tf


def f(x, flag=1):
    if flag:
        return tf.reduce_sum(x)
    return tf.reduce_sum(x) * 2


f(tf.constant([1.0, 2.0]))
