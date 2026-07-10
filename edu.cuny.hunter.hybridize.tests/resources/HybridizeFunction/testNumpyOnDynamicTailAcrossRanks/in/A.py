import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def tail_prod(x):
    inner = np.prod(get_shape(x)[-1:])
    return tf.reshape(x, [-1, inner])


# The two feeds disagree on rank, so the shape vector's rank is statically
# unresolvable, but the suffix slice covers "the last dimension" regardless:
# the rank-3 feed's trailing dimension is static while the rank-2 feed's is
# dynamic, so numpy over it consumes a provably-dynamic dimension.
cube = tf.ones((2, 3, 4))
dyn = tf.reshape(tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]])), [2, -1])

if tf.executing_eagerly():
    picked = cube
else:
    picked = dyn

assert tail_prod(picked).shape == (6, 4)
