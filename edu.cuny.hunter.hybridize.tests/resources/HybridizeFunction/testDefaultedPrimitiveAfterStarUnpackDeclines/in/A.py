# #795 with the wala/ML#751 star-argument fix, mirroring the corpus `compute_loss` case. `combine`'s `scale`
# is a defaulted primitive supplied positionally by the trailing `i` (a varying loop index), so it is a
# retrace key and `combine` should decline. The `*rest` unpack fills two parameters (`b`, `c`) but occupies a
# single invoke slot, so whether the trailing `i` reaches `scale` is undetermined. Reading the invoke's
# starred positions (wala/ML#751), the supplied-parameter analysis returns null (not a definite unsupplied)
# for `scale`, so the defaulted-primitive exemption does not apply and `combine` correctly declines.
import tensorflow as tf


def combine(a, b, c, scale=1):
    return tf.reduce_sum(a) * scale


def driver(rest):
    for i in range(3):
        combine(tf.constant([1.0, 2.0]), *rest, i)


driver([tf.constant([3.0, 4.0]), tf.constant([5.0, 6.0])])
