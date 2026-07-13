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


# Each function's feeds disagree on rank (the rank-3 `cube` and a rank-2
# `tf.keras.Input`), so the shape vector's rank is statically unresolvable. The
# rank-2 feed's axes are graph-time `None` (dynamic evidence), so the suffix,
# prefix, and copy slices — whose coverage is rank-independent — cover a
# provably-dynamic dimension; `pair_prod`'s [-3:] additionally clamps past the
# rank-2 feed's extent. The empty slice covers nothing, so numpy over it is
# vacuously safe. The strided and mid-start slices have rank-dependent coverage,
# so their provenance is untracked and unprovable. A compile-time-constant feed
# would instead let Ariadne fold the rank-2 extents (wala/ML#722).
cube = tf.ones((2, 3, 4))
tail_dyn = tf.keras.Input(shape=(None,))
head_dyn = tf.keras.Input(shape=(2,))

if tf.executing_eagerly():
    tail_pick = cube
    head_pick = cube
    copy_pick = cube
else:
    tail_pick = tail_dyn
    head_pick = head_dyn
    copy_pick = tail_dyn

tail_prod(tail_pick)
head_prod(head_pick)
copy_prod(copy_pick)
pair_prod(copy_pick)
nil_prod(copy_pick)
stride_prod(copy_pick)
rest_prod(copy_pick)
