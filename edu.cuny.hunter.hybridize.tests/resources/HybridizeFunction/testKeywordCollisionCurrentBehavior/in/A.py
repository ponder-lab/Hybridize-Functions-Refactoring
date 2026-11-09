# #791 marker for the trampoline keyword-name collision fixed upstream in wala/ML#740 (released in
# Ariadne 0.52.34). `M.check` is reached from two call sites that collide on total arity and differ
# only by keyword (`step` vs `mode`). Through 0.52.33, trampoline bodies were keyed on
# (receiver, total argument count), so both call sites shared one body whose named slots came from
# whichever populated first, and the other call's keyword never reached its parameter. 0.52.34 keys on
# the keyword-name set as well, so both keywords now bind.
import tensorflow as tf


class M:
    def check(self, x, mode="v", step=None):
        if mode == "v":
            return tf.reduce_sum(x)
        return tf.reduce_sum(x) * 2


m = M()
t = tf.constant([1.0, 2.0])
m.check(t, step=1)
m.check(t, mode="d")
