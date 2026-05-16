# Regression fixture for #498. The `__call__` method on a `tf.keras.Model` subclass is
# called with a non-tensor (`5`). No parameter classifies as tensor-typed on its own (no
# type hint, no Ariadne call-site classification, no container). But the function name
# matches the speculative-context regex (`call|__call__|...`) AND the owning class
# inherits from `tf.keras.Model`, so `Function.inferTensorParameters`' speculative-context
# branch fires and sets `hasTensorParameter = TRUE` without any parameter having
# `isTensor() == TRUE`. Pins the asymmetry: `getHasTensorParameter() == TRUE` does NOT
# imply `∃ non-self param p : p.isTensor() == TRUE`.

import tensorflow as tf


class M(tf.keras.Model):
    def __call__(self, x):
        return x


M()(5)
