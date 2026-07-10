import numpy as np
import tensorflow as tf


def get_shape(tensor):
    return tensor.shape.as_list()


def tail_prod(x):
    inner = np.prod(get_shape(x)[-1:])
    return tf.reshape(x, [-1, inner])


def head_prod(x):
    outer = np.prod(get_shape(x)[:1])
    return tf.reshape(x, [outer, -1])


def copy_prod(x):
    full = np.prod(get_shape(x)[:])
    return tf.reshape(x, [full])


def pair_prod(x):
    inner = np.prod(get_shape(x)[-3:])
    return tf.reshape(x, [-1, inner])


def nil_prod(x):
    unused = np.prod(get_shape(x)[:0])
    return tf.reshape(x, [-1, 1])


def stride_prod(x):
    inner = np.prod(get_shape(x)[::2])
    return tf.reshape(x, [-1, inner])


def rest_prod(x):
    inner = np.prod(get_shape(x)[1:])
    return tf.reshape(x, [-1, inner])


# Each function's feeds disagree on rank, so the shape vector's rank is
# statically unresolvable. The suffix, prefix, and copy slices have
# rank-independent coverage and track: the rank-2 feed makes a covered
# dimension provably dynamic, so numpy over them consumes a dynamic
# dimension; `pair_prod`'s [-3:] additionally clamps past the rank-2 feed's
# extent. The empty slice covers nothing, so numpy over it is vacuously
# safe. The strided and mid-start slices have rank-dependent coverage, so
# their provenance is untracked and unprovable.
cube = tf.ones((2, 3, 4))
tail_dyn = tf.reshape(tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]])), [2, -1])
head_dyn = tf.reshape(tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]])), [-1, 2])

if tf.executing_eagerly():
    tail_pick = cube
    head_pick = cube
    copy_pick = cube
else:
    tail_pick = tail_dyn
    head_pick = head_dyn
    copy_pick = tail_dyn

assert tail_prod(tail_pick).shape == (6, 4)
assert head_prod(head_pick).shape == (2, 12)
assert copy_prod(copy_pick).shape == (24,)
assert pair_prod(copy_pick).shape == (1, 24)
assert nil_prod(copy_pick).shape == (24, 1)
assert stride_prod(copy_pick).shape == (3, 8)
assert rest_prod(copy_pick).shape == (2, 12)
