import tensorflow as tf


# The `wild_*` arms are called with tensors of differing rank, so inference degrades their
# parameter's shape to unknown rank (`shape=None`); each body then reads the input's static
# rank surface (`as_list`, `len(shape)`, `rank`/`ndims`), which stops working once the
# parameter is pinned to unknown rank. The `ranked_*` twins are called at a consistent rank
# with differing extents, so their specs keep a known rank with a dynamic axis, which every
# rank read tolerates: only the wild arms may block.
def wild_as_list(x):
    return tf.ones(x.shape.as_list()[-1])


def ranked_as_list(x):
    return tf.ones(x.shape.as_list()[-1])


def wild_len(x):
    if len(x.shape) == 1:
        return tf.reshape(x, [-1, 1])
    return x


def ranked_len(x):
    if len(x.shape) == 2:
        return x
    return tf.reshape(x, [-1, 1])


def wild_rank(x):
    r = x.shape.rank
    return tf.ones(r + 1)


# Arithmetic over a known rank is a trace-time constant even when an axis is dynamic; this
# arm pins the precision direction (it must NOT block).
def ranked_rank(x):
    r = x.shape.rank
    return tf.ones(r + 1)


def wild_ndims(x):
    return tf.ones(x.shape.ndims)


v = tf.ones((3,))
m = tf.ones((2, 3))
m2 = tf.ones((4, 3))

assert wild_as_list(v).shape == (3,)
assert wild_as_list(m).shape == (3,)
assert ranked_as_list(m).shape == (3,)
assert ranked_as_list(m2).shape == (3,)
assert wild_len(v).shape == (3, 1)
assert wild_len(m).shape == (2, 3)
assert ranked_len(m).shape == (2, 3)
assert ranked_len(m2).shape == (4, 3)
assert wild_rank(v).shape == (2,)
assert wild_rank(m).shape == (3,)
assert ranked_rank(m).shape == (3,)
assert ranked_rank(m2).shape == (3,)
assert wild_ndims(v).shape == (1,)
assert wild_ndims(m).shape == (2,)
