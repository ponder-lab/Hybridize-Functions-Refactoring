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


class Masker:
    @classmethod
    def trim(cls, x, name=None):
        return tf.boolean_mask(x, tf.reduce_sum(x, 1) > 0.0, name=name)

    def trim_this(this, x, name=None):
        return tf.boolean_mask(x, tf.reduce_sum(x, 1) > 0.0, name=name)


def classmethod_positional(x):
    return Masker.trim(x, "trimmed")


def classmethod_omitted(x):
    return Masker.trim(x)


def this_positional(x):
    return Masker().trim_this(x, "trimmed")


def this_omitted(x):
    return Masker().trim_this(x)


def unbound_positional(x):
    return Masker.trim_this(Masker(), x, "trimmed")


def unbound_omitted(x):
    return Masker.trim_this(Masker(), x)


def row_sum(row):
    return tf.reduce_sum(trim_zeros(tf.reshape(row, [1, 4]))[0])


def maps_caller(x):
    return tf.map_fn(row_sum, x)


def row_mask(row, name=None):
    return tf.reduce_sum(tf.boolean_mask(row, row > 0.0, name=name))


def maps_forwarder(x):
    return tf.map_fn(row_mask, x)


def forwards(x, name=None):
    return trim_zeros(x, name=name)[0]


def calls_forwards(x):
    return forwards(x) * 2.0


b = tf.constant([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
assert trim_zeros(b)[0].shape == (1, 4)
assert evaluate(b).shape == (2, 4)
assert trims(b).shape == (1, 4)
assert evaluates(b).shape == (2, 4)
assert calls_trims(b).shape == (1, 4)
assert trims_named(b).shape == (1, 4)
assert trims_positional(b).shape == (1, 4)
assert classmethod_positional(b).shape == (1, 4)
assert classmethod_omitted(b).shape == (1, 4)
assert this_positional(b).shape == (1, 4)
assert this_omitted(b).shape == (1, 4)
assert unbound_positional(b).shape == (1, 4)
assert unbound_omitted(b).shape == (1, 4)
assert maps_caller(b).shape == (2,)
assert maps_forwarder(b).shape == (2,)
assert calls_forwards(b).shape == (1, 4)
