# Fixture for #1323: a static method's decorator changes how a call binds its arguments, so it
# declines as any decorated function does.
import tensorflow as tf


class Ops:

    @staticmethod
    def f(x, axis=-1):
        return tf.reduce_sum(x, axis=axis)


Ops.f(tf.constant([1.0, 2.0]), -1)
Ops().f(tf.constant([3.0, 4.0]), axis=-1)
