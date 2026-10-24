import numpy as np
import tensorflow as tf


def g(x):
    inner = np.prod(tf.shape(x)[-1:])
    return tf.reshape(x, [-1, inner])


g(tf.ones((3, 4)))
