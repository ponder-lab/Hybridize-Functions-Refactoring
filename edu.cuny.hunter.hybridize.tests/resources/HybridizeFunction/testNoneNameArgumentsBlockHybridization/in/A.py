import tensorflow as tf

MASK_NAME = "kept"


def trim_zeros(boxes, name=None):
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def mask_none(x):
    return tf.boolean_mask(x, tf.reduce_sum(x, 1) > 0.0, name=None)


def transpose_none(x):
    return tf.linalg.matrix_transpose(x, name=None)


def sqrt_none(x):
    return tf.sqrt(x, name=None)


def mask_named(x, name="kept"):
    return tf.boolean_mask(x, tf.reduce_sum(x, 1) > 0.0, name=name)


def mask_constant(x, name=MASK_NAME):
    return tf.boolean_mask(x, tf.reduce_sum(x, 1) > 0.0, name=name)


def mask_default(x):
    return tf.boolean_mask(x, tf.reduce_sum(x, 1) > 0.0)


def trims(boxes):
    return trim_zeros(boxes)[0]


b = tf.constant([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
assert trim_zeros(b)[0].shape == (1, 4)
assert mask_none(b).shape == (1, 4)
assert transpose_none(b).shape == (4, 2)
assert sqrt_none(b).shape == (2, 4)
assert mask_named(b).shape == (1, 4)
assert mask_constant(b).shape == (1, 4)
assert mask_default(b).shape == (1, 4)
assert trims(b).shape == (1, 4)
