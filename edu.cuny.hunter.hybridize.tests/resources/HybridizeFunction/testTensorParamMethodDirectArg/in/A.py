import tensorflow as tf


# Delta vs testParameterTypedFromArgumentNotUse: f is a METHOD (self offset), arg still direct.
# `x` should be tensor-typed from the direct tensor argument at the call site below.
class M:
    def f(self, x):
        return tf.linalg.matmul(x, x)


M().f(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
