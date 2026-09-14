import numpy as np
import tensorflow as tf


# The outer slice's receiver is the INNER slice's RESULT, so the taint reaches numpy only if
# coloring the inner slice's def re-enters the worklist and re-matches on the outer slice.
# `testNumpyOnSlicedTensorValue` and `testNumpyOnSlicedTensorArithmetic` both slice a bare
# parameter exactly once, so neither exercises the chain.
def chained_slice(bboxA):
    inner = bboxA[:, 1:]
    outer = inner[:, :2]
    return np.maximum(outer, 0.0)


b = tf.ones((4, 4))
chained_slice(b)
