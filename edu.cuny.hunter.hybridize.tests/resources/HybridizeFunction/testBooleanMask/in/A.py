# From: https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/boolean_mask#examples

import tensorflow as tf
import numpy as np


def f(a):
    pass


tensor = [0, 1, 2, 3]  # 1-D example
mask = np.array([True, False, True, False])
r = tf.boolean_mask(tensor, mask)
f(r)
