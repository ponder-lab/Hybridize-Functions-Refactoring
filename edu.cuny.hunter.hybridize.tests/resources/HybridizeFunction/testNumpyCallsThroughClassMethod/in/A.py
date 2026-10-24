import numpy as np
import tensorflow as tf


class C:

    @classmethod
    def k(cls, x):
        return np.asarray(x)

    @classmethod
    def j(cls, n, x):
        return x * np.asarray(n)


def f(a):
    return C.k(a)


def g(a):
    return C.j(3, a)


t = tf.ones((3, 4))
f(t)
g(t)
