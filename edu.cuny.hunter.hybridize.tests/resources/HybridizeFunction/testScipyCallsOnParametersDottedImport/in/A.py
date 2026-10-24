import scipy.special
import tensorflow as tf


def f(a):
    return scipy.special.expit(a)


t = tf.ones((3, 4))
f(t)
