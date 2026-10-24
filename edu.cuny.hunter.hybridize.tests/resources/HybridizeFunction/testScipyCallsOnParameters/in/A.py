import scipy
import tensorflow as tf
from scipy import special


def f(a):
    return special.expit(a)


def g(a):
    return scipy.special.expit(a)


t = tf.ones((3, 4))
f(t)
g(t)
