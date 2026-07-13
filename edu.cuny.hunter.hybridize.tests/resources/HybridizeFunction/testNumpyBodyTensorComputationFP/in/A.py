import numpy as np
import tensorflow as tf


def numpy_body(x):
    m = np.zeros(x.shape)
    return np.array(m, dtype=np.int32)


def tensor_body(x):
    return tf.reduce_sum(x)


a = np.array([1, 2, 3])
numpy_body(a)
tensor_body(a)
