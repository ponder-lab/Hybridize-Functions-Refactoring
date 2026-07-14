import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def op_body(x):
    m = np.zeros(x.shape)
    return m + m


def subscript_write_body(x):
    m = np.zeros(x.shape)
    m[0] = 1.0
    return m


def subscript_read_body(x):
    return x[0]


def helper(x):
    return np.array(x, dtype=np.int32)


def interproc_body(x):
    return helper(x)


def scipy_body(x):
    return sp.diags(x)


def reshape_method_body(x):
    m = np.array(x, dtype=np.int32)
    return m.reshape((1, -1))


def tf_control(x):
    return tf.reduce_sum(x)


a = np.array([1, 2, 3])
op_body(a)
subscript_write_body(a)
subscript_read_body(a)
interproc_body(a)
scipy_body(a)
reshape_method_body(a)
tf_control(a)
