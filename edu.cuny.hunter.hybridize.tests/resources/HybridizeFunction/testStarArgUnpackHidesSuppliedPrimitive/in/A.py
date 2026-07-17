# #795 known limitation, mirroring the corpus `compute_loss` false exemption. `combine`'s `scale` is a
# defaulted primitive supplied positionally by the trailing `i` (a varying loop index), so it is a retrace
# key and `combine` should decline. The supplied-parameter analysis misses it: the `*rest` unpack fills two
# parameters (`b`, `c`) but occupies a single invoke slot, so the trailing `i` lands one slot short of
# `scale`'s expected position and the positional-count test reads `scale` as unsupplied. `scale` is then
# wrongly exempted and `combine` is wrongly optimizable. This pins the current (incorrect) behavior; it
# inverts once the invoke exposes star-argument presence so the analysis can decline to align (wala/ML#751).
import tensorflow as tf


def combine(a, b, c, scale=1):
    return tf.reduce_sum(a) * scale


def driver(rest):
    for i in range(3):
        combine(tf.constant([1.0, 2.0]), *rest, i)


driver([tf.constant([3.0, 4.0]), tf.constant([5.0, 6.0])])
