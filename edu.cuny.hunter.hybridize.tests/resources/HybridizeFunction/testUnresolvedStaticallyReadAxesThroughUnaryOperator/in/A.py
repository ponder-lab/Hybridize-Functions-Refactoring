import tensorflow as tf


def negated(x):
    return tf.range(-x.shape[-1], 0)


def negated_unused(x):
    m = -x.shape[-1]
    return x * 2.0


def positive_unused(x):
    m = +x.shape[-1]
    return x * 2.0


def inverted_unused(x):
    m = ~x.shape[-1]
    return x * 2.0


def absent(x):
    empty = not x.shape[-1]
    return tf.where(empty, tf.zeros_like(x), x)


def suffix_literal(x):
    dims = x.shape[-1:]
    return tf.range(dims[0] - 1, -1, -1)


def suffix_variable(x):
    k = 1
    dims = x.shape[-k:]
    return tf.range(dims[0] - 1, -1, -1)


def prefix_variable(x):
    k = 1
    dims = x.shape[:k]
    return tf.range(dims[0] - 1, -1, -1)


a = tf.ones((2, 4))
b = tf.ones((2, 5))

assert negated(a).shape == (4,)
assert negated(b).shape == (5,)
assert negated_unused(a).shape == (2, 4)
assert negated_unused(b).shape == (2, 5)
assert positive_unused(a).shape == (2, 4)
assert positive_unused(b).shape == (2, 5)
assert inverted_unused(a).shape == (2, 4)
assert inverted_unused(b).shape == (2, 5)
assert absent(a).shape == (2, 4)
assert absent(b).shape == (2, 5)
assert suffix_literal(a).shape == (4,)
assert suffix_literal(b).shape == (5,)
assert suffix_variable(a).shape == (4,)
assert suffix_variable(b).shape == (5,)
assert prefix_variable(a).shape == (2,)
assert prefix_variable(b).shape == (2,)
