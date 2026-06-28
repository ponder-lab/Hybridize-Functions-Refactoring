import tensorflow as tf


# Delta vs testParameterTypedFromArgumentNotUse: still a module function, but the argument is a
# DERIVED tensor (g's result), not a direct tf.constant. `x` should be tensor-typed.
def g(y):
    return y * 2.0


def f(x):
    return tf.linalg.matmul(x, x)


f(g(tf.constant([[1.0, 2.0], [3.0, 4.0]])))
