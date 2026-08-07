import numpy as np
import tensorflow as tf

T = np.array([1.5, 2.5])
P = tf.ones([2])


def loss(target, pred):
    return tf.abs(target - pred)


# Reduces TensorFlow2.0-Examples' RPN `compute_loss` (#878): the subtraction's two
# operands are both parameters, so each one's only partner operand is the other, and
# deciding either dtype from the other is circular. Eagerly the float64 NumPy argument
# is converted at the subtraction under the float32 tensor operand's dtype; the detector
# must impose nothing here rather than let the pair swap dtypes.
loss(T, P)
