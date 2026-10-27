import numpy as np
import tensorflow as tf


# Reduces TensorFlow2.0-Examples' FPN `_upsample_add` (#882 Case 1): tuple-unpacking the
# shape reads its axes, and the extents feed `tf.image.resize`'s `size`, which requires
# Python integers. `wild_upsample`'s second argument arrives with differing extents on
# axis 1, so its spec leaves that axis wild and `H` is None at trace time; the ranked
# twin's extents agree, so every read axis is concrete and the spec is writable.
def wild_upsample(x, y):
    _, H, W, C = y.shape
    return tf.image.resize(x, size=(H, W), method="bilinear")


def ranked_upsample(x, y):
    _, H, W, C = y.shape
    return tf.image.resize(x, size=(H, W), method="bilinear")


# Iterating the shape raises on unknown rank ("Cannot iterate over a shape with unknown
# rank"); `wild_iter` is called at differing ranks, degrading its spec to `shape=None`,
# while `ranked_iter` keeps a known rank with a dynamic axis, which iteration tolerates.
def wild_iter(x):
    last = None
    for d in x.shape:
        last = d
    return tf.abs(x)


def ranked_iter(x):
    last = None
    for d in x.shape:
        last = d
    return tf.abs(x)


# Reduces deep_recommenders' transformer position encoding (#882 Case 2): a NumPy buffer
# sized from a statically-read axis requires a Python integer, which a wildcard's None is
# not. `np_sized` reads the wild axis 1; `ranked_np` reads the concrete axis 2.
def np_sized(x):
    n = x.shape[1]
    pos = np.zeros((n, 4))
    return x + tf.cast(pos, tf.float32)


def ranked_np(x):
    m = x.shape[2]
    pos = np.zeros((m,))
    return x + tf.cast(pos, tf.float32)


img = tf.ones((1, 8, 8, 3))
y4 = tf.ones((1, 4, 4, 3))
y6 = tf.ones((1, 6, 4, 3))

assert wild_upsample(img, y4).shape == (1, 4, 4, 3)
assert wild_upsample(img, y6).shape == (1, 6, 4, 3)
assert ranked_upsample(img, y4).shape == (1, 4, 4, 3)
assert ranked_upsample(img, y4).shape == (1, 4, 4, 3)

v = tf.ones((3,))
m = tf.ones((2, 4))
m2 = tf.ones((2, 6))

assert wild_iter(v).shape == (3,)
assert wild_iter(m).shape == (2, 4)
assert ranked_iter(m).shape == (2, 4)
assert ranked_iter(m2).shape == (2, 6)

t4 = tf.ones((2, 4, 4))
t6 = tf.ones((2, 6, 4))

assert np_sized(t4).shape == (2, 4, 4)
assert np_sized(t6).shape == (2, 6, 4)
assert ranked_np(t4).shape == (2, 4, 4)
assert ranked_np(t6).shape == (2, 6, 4)
