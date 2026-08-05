# #795 tripwire for the wala/ML#743 (0.52.35) regression. `C.m`'s `flag` is a defaulted primitive
# no call site supplies. On 0.52.34 the trampolined method's default is not materialized, so `flag`
# has an empty points-to set and never triggers HAS_PRIMITIVE_PARAMETERS (this test passes via that
# path). At 0.52.35, wala/ML#743 materializes the default, so `flag` becomes a detected primitive;
# without #795's exemption this test flips red (m declines), and the exemption restores it.
import tensorflow as tf


class C:

    def m(self, x, flag=1):
        if flag:
            return tf.reduce_sum(x)
        return tf.reduce_sum(x) * 2


c = C()
c.m(tf.constant([1.0, 2.0]))
