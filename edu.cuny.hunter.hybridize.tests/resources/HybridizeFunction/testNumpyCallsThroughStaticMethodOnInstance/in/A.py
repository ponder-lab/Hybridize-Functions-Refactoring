import numpy as np
import tensorflow as tf


class D:

    @staticmethod
    def s(x):
        return np.asarray(x)


def h(a):
    o = D()
    return o.s(a)


t = tf.ones((3, 4))
h(t)
