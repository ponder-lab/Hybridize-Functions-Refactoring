# Fixture for #787 exercising the trampoline path, which the module-level-function fixtures do
# not reach. Ariadne interposes a synthesized trampoline between a caller and an instance method
# to bind `self`, so this method's call-graph predecessor is that trampoline rather than the
# module code below. The supply check reads the forwarded invoke's arity, which the trampoline
# copies from the originating call, so `training` must still come back as supplied by nobody.
#
# This is the shape that matters in real model code: `call(self, x, training=True)` is the Keras
# idiom, and a defaulted `training` is by far the most common non-tensor parameter in practice.
import tensorflow as tf


class Layer:

    def call(self, x, training=True):
        if training:
            return tf.reduce_sum(x)
        return tf.reduce_sum(x) * 2


layer = Layer()
layer.call(tf.constant([1.0, 2.0]))
