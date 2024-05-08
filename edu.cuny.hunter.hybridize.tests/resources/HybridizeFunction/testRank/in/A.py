# https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/rank#for_example

import tensorflow as tf


def f(a):
    pass


# shape of tensor 't' is [2, 2, 3]
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
r = tf.rank(t)  # 3
f(r)
