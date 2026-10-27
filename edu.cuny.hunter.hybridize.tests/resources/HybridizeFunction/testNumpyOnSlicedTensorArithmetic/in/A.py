import numpy as np
import tensorflow as tf


# The slice result is consumed by ARITHMETIC first, and numpy is applied to the arithmetic
# result, so no slice is a direct operand of the numpy call. `testNumpyOnSlicedTensorValue`
# pins the direct-operand shape; this pins the arithmetic-mediated one, which that fixture
# does not reach.
def rerec_arith(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    return np.maximum(w, h)


b = tf.ones((4, 4))
rerec_arith(b)
