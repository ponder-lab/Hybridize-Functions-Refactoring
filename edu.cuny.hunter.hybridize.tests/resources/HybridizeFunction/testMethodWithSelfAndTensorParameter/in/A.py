# Regression fixture for #498. A method `m` with `self` and a tensor parameter `t`. After
# `Parameter.classifyAsTensor` runs on each, `self.isTensor() == FALSE` (self-check), and
# `t.isTensor() == TRUE` (Ariadne classifies from the tensor call site). Pins that
# classification is per-parameter and works correctly across method-style call patterns.

import tensorflow as tf


class C:
    def m(self, t):
        return t


C().m(tf.constant([1.0, 2.0]))
