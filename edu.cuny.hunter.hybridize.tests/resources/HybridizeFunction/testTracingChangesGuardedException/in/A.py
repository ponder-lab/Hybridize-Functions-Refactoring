import unittest

import pytest
import tensorflow as tf

# Pins #1014: a call declared to fail with a TensorFlow error can raise another exception once the
# function is hybridized, since a bare decorator traces it with the argument's own shape and dtype,
# and an operation whose static check fails on them raises at trace time instead of at run time.
# Each refused function's guarded call raises what its guard declares eagerly but not under a bare
# decorator, and each converted one's raises it under both, on the pinned TensorFlow 2.9.3. Each
# function's conforming call precedes its guard.

case = unittest.TestCase()


# Refused: a rank-1 argument fails `tf.matmul`'s static shape check, so tracing raises ValueError.
def rank_mismatch(x):
    return tf.matmul(x, x)


rank_mismatch(tf.ones((2, 2)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    rank_mismatch(tf.ones((3,)))


# Refused: the pytest spelling of the same shape.
def pytest_rank_mismatch(x):
    return tf.matmul(x, x)


pytest_rank_mismatch(tf.ones((2, 2)))

with pytest.raises(tf.errors.InvalidArgumentError):
    pytest_rank_mismatch(tf.ones((3,)))


# Converted: an argument of the conforming call's shape traces cleanly, and the assertion fails in
# the kernel.
def same_shape_assert(x):
    tf.debugging.assert_positive(x)
    return x * 2


same_shape_assert(tf.ones((3,)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    same_shape_assert(tf.zeros((3,)))


# Converted: a parameter that is not a tensor, here left at its default, is not compared.
def same_shape_default(x, scale=None):
    tf.debugging.assert_positive(x)
    return x * 2


same_shape_default(tf.ones((3,)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    same_shape_default(tf.zeros((3,)))


# Converted: the index is out of range only by its value, which the kernel checks.
def same_shape_gather(x, i):
    return tf.gather(x, i)


same_shape_gather(tf.ones((5,)), tf.constant(1))

with case.assertRaises(tf.errors.InvalidArgumentError):
    same_shape_gather(tf.ones((5,)), tf.constant(7))


# Refused: the same assertion on a shape no other call passes. The failure is data-dependent and
# survives a bare decorator, but nothing tells it from a static one.
def shape_differs_assert(x):
    tf.debugging.assert_positive(x)
    return x * 2


shape_differs_assert(tf.ones((3,)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    shape_differs_assert(tf.zeros((2,)))


# Converted: the guard admits the ValueError tracing raises.
def admits_value(x):
    return tf.matmul(x, x)


admits_value(tf.ones((2, 2)))

with case.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
    admits_value(tf.ones((3,)))


# Converted: `Exception` admits any exception.
def admits_any(x):
    return tf.matmul(x, x)


admits_any(tf.ones((2, 2)))

with case.assertRaises(Exception):
    admits_any(tf.ones((3,)))


# Refused: an argument of another dtype fails `tf.sqrt`'s static dtype check, so tracing raises
# TypeError, which a guard admitting ValueError does not admit.
def dtype_mismatch(x):
    return tf.sqrt(x)


dtype_mismatch(tf.ones((2,)))

with case.assertRaises((tf.errors.InvalidArgumentError, ValueError)):
    dtype_mismatch(tf.ones((2,), tf.int32))


# Converted: the guard admits the TypeError tracing raises.
def admits_type(x):
    return tf.sqrt(x)


admits_type(tf.ones((2,)))

with case.assertRaises((tf.errors.InvalidArgumentError, TypeError)):
    admits_type(tf.ones((2,), tf.int32))


# Refused: the shape of a dict's element is not the argument's, so it is not known to match.
def from_dict(features):
    return tf.matmul(features["a"], features["a"])


from_dict({"a": tf.ones((2, 2))})

with case.assertRaises(tf.errors.InvalidArgumentError):
    from_dict({"a": tf.ones((3,))})


# Refused: every call is declared to fail, so no call is known to trace the argument cleanly.
def only_declared(x):
    return tf.matmul(x, x)


with case.assertRaises(tf.errors.InvalidArgumentError):
    only_declared(tf.ones((3,)))


# Converted: the body raises OutOfRangeError itself, where tracing raises it too, and a failed static
# check is raised as InvalidArgumentError instead, which the guard does not admit.
def out_of_range_guard(x):
    if x.shape[0] == 3:
        raise tf.errors.OutOfRangeError(None, None, "exhausted")
    return x * 2


out_of_range_guard(tf.ones((2,)))

with case.assertRaises(tf.errors.OutOfRangeError):
    out_of_range_guard(tf.ones((3,)))


# Converted: a guard naming no `tf.errors` class declares an exception tracing raises where the body
# does.
def no_op_error(x):
    return x * 2


no_op_error(tf.ones((2,)))

with case.assertRaises(TypeError):
    no_op_error(None)


# Unaffected: an already hybrid function is not converted, so the decorator it has changes nothing.
@tf.function
def already_hybrid(x):
    tf.debugging.assert_positive(x)
    return x * 2


already_hybrid(tf.ones((3,)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    already_hybrid(tf.zeros((2,)))


# Refused: both calls pass a tensor whose leading extent depends on a mask's values, so the type the
# conforming call passes is not known to be the one the declared failure passes.
def masked(x):
    return tf.matmul(x, x)


masked(tf.boolean_mask(tf.ones((2, 2)), [True, True]))

with case.assertRaises(tf.errors.InvalidArgumentError):
    masked(tf.boolean_mask(tf.ones((3, 2)), [True, True, True]))
