import tensorflow as tf


def trim_zeros(boxes, name=None):
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def evaluate(x):
    return tf.sqrt(x, tf.float32)


def trims(boxes):
    return trim_zeros(boxes)[0]


def evaluates(x):
    return evaluate(x) + 1.0


def calls_trims(boxes):
    return trims(boxes) * 2.0


def trims_named(boxes):
    return trim_zeros(boxes, name="trimmed")[0]


def trims_positional(boxes):
    return trim_zeros(boxes, "trimmed")[0]


b = tf.constant([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
assert trim_zeros(b)[0].shape == (1, 4)
assert evaluate(b).shape == (2, 4)
assert trims(b).shape == (1, 4)
assert evaluates(b).shape == (2, 4)
assert calls_trims(b).shape == (1, 4)
assert trims_named(b).shape == (1, 4)
assert trims_positional(b).shape == (1, 4)
