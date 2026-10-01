import unittest

import pytest
import tensorflow as tf

# Pins #1005: a call declared to fail is set aside as evidence (#888), but it still runs against an
# inferred signature, which validates the argument before the body does. Where the signature would
# reject that argument with an exception the guard does not admit, emitting it would make the test
# fail, so the signature is withheld. Each function's conforming call precedes its guard, since a
# guard dominates every later call in the same frame. Every guarded call raises what its guard
# declares on the pinned TensorFlow 2.9.3, eagerly and under a bare decorator.

case = unittest.TestCase()


# Withheld: a TensorSpec rejects None with ValueError, which `TypeError` does not admit.
def type_guard(x):
    return x * 2


type_guard(tf.ones([2]))

with case.assertRaises(TypeError):
    type_guard(None)


# Kept: the body raises ValueError on the wrong extent, and so does the signature's rejection.
def value_guard(x):
    if x.shape[0] != 2:
        raise ValueError("expected two")
    return x * 2


value_guard(tf.ones([2]))

with case.assertRaises(ValueError):
    value_guard(tf.ones([3]))


# Kept: a tuple of classes admits the signature's ValueError.
def tuple_guard(x):
    return x * 2


tuple_guard(tf.ones([2]))

with case.assertRaises((TypeError, ValueError)):
    tuple_guard(None)


# Withheld: the pytest spelling, with the same rejection as `type_guard`.
def pytest_type_guard(x):
    return x * 2


pytest_type_guard(tf.ones([2]))

with pytest.raises(TypeError):
    pytest_type_guard(None)


# Kept: `Exception` admits any rejection.
def broad_guard(x):
    return x * 2


broad_guard(tf.ones([2]))

with case.assertRaises(Exception):
    broad_guard(None)


# Withheld: the class is the guard's first argument, not the pattern after it.
def regex_guard(x):
    return x * 2


regex_guard(tf.ones([2]))

with case.assertRaisesRegex(TypeError, "unsupported"):
    regex_guard(None)


# Kept: the declared failure passes a conforming tensor, so the signature does not intervene.
STRICT = [False]


def conforming(x):
    if STRICT[0]:
        raise TypeError("strict")
    return x * 2


conforming(tf.ones([2]))
STRICT[0] = True

with case.assertRaises(TypeError):
    conforming(tf.ones([2]))

STRICT[0] = False


# Withheld, and refused (#1014): the signature rejects the rank-1 argument with ValueError, and so
# does the bare decorator, from the static shape check at trace time.
def op_error_guard(x):
    return tf.matmul(x, x)


op_error_guard(tf.ones((2, 2)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    op_error_guard(tf.ones((3,)))


# Kept: a nested specification rejects a non-container with TypeError, which the guard admits.
def nested_type_guard(pair):
    return pair[0] + pair[1]


nested_type_guard((tf.ones([2]), tf.ones([2])))

with case.assertRaises(TypeError):
    nested_type_guard(None)


# Withheld: a flat specification converts a list of tensors of different shapes and raises
# InvalidArgumentError, which `TypeError` does not admit.
def list_type_guard(x):
    return x * 2.0


list_type_guard(tf.ones([2, 3]))

with case.assertRaises(TypeError):
    list_type_guard([tf.ones([2]), tf.ones([3])])


# Withheld: the same conversion raises InvalidArgumentError, not the ValueError the guard admits, so
# a container argument's rejection is not predicted as ValueError.
def list_value_guard(x):
    if isinstance(x, list):
        raise ValueError("a list")
    return x * 2.0


list_value_guard(tf.ones([2, 3]))

with case.assertRaises(ValueError):
    list_value_guard([tf.ones([2]), tf.ones([3])])


# Kept: a nested specification rejects a container of another structure with ValueError, which the
# guard admits.
def nested_value_guard(pair):
    if len(pair) != 2:
        raise ValueError("a pair")
    return pair[0] + pair[1]


nested_value_guard((tf.ones([2]), tf.ones([2])))

with case.assertRaises(ValueError):
    nested_value_guard((tf.ones([2]), tf.ones([2]), tf.ones([2])))


# Withheld: a data-dependent op error survives a bare decorator, but the signature rejects the
# out-of-extent argument with ValueError first. Refused too (#1014), since no other call passes that
# shape, so nothing tells the error from a static one.
def gather_oob(x):
    return tf.gather(x, 3)


gather_oob(tf.ones([5]))

with case.assertRaises(tf.errors.InvalidArgumentError):
    gather_oob(tf.ones([2]))


# Withheld: a sparse specification rejects None with TypeError, which `ValueError` does not admit.
def sparse_value(x):
    if x is None:
        raise ValueError("no input")
    return tf.sparse.reduce_sum(x)


sparse_value(tf.sparse.from_dense(tf.ones([2])))

with case.assertRaises(ValueError):
    sparse_value(None)


# Withheld: the guarded call passes a tensor and then None, so its argument may be either.
def mixed(x):
    return x * 2


mixed(tf.ones([2]))

with case.assertRaises(TypeError):
    for a in (tf.ones([2]), None):
        mixed(a)


# Withheld: a dict with string keys against a nested specification raises KeyError.
def pair_dict(pair):
    if isinstance(pair, dict):
        raise ValueError("a dict")
    return pair[0] + pair[1]


pair_dict((tf.ones([2]), tf.ones([2])))

with case.assertRaises(ValueError):
    pair_dict({"a": tf.ones([2]), "b": tf.ones([2])})
