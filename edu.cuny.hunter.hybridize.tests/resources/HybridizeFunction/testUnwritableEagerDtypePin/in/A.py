import numpy as np
import tensorflow as tf

V = tf.Variable(tf.zeros([2]), name="weight")

X64 = np.array([1.5, 2.5])
X32 = np.array([1.5, 2.5], dtype=np.float32)


def changed(x):
    return tf.reduce_sum(V * x)


def unchanged(x):
    return tf.reduce_sum(V * x)


def upstream(v):
    tf.reduce_sum(V * v)
    return v


def forwarded(x):
    return tf.reduce_sum(V * x)


# The fed dtype differs from the one the multiply imposes, so eager execution converts the
# argument at the operation. Without a specification to reproduce that conversion at the
# boundary, tracing materializes the argument as float64 and the multiply raises.
changed(X64)


# The same operation, fed the dtype it imposes. Nothing is converted, so a bare decorator
# carries the argument in unchanged and the function trace matches the eager run.
unchanged(X32)


# The argument is a parameter that was itself coerced upstream and then forwarded
# unchanged, so the fed side reaching this parameter carries both dtypes: the analysis
# holds the one imposed there, while the run forwards the original. That is a disagreement
# rather than an absence of evidence, and it reads as a change however incomplete the fed
# side is. Executed, this raises bare exactly as the first one does, so the reading is
# right. Forwarding the coerced parameter is what creates the state; returning a product of
# it would hand on a fresh value whose dtype both sides agree on.
forwarded(upstream(X64))
