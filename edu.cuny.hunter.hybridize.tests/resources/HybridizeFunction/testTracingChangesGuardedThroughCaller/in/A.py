import unittest

import tensorflow as tf

# A guard around a call reaches the function the call reaches through a user function between them,
# since what the function raises leaves that frame unhandled. Each refused function's guarded call
# raises what its guard declares eagerly but not under a bare decorator, and the converted one's
# raises it under both, on the pinned TensorFlow 2.9.3. Each function's conforming call precedes
# its guard.

case = unittest.TestCase()


# Refused: a rank-1 argument fails `tf.matmul`'s static shape check, so tracing raises ValueError,
# which leaves `call_forwarded` and escapes the guard.
def forwarded(x):
    return tf.matmul(x, x)


def call_forwarded(x):
    return forwarded(x)


forwarded(tf.ones((2, 2)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    call_forwarded(tf.ones((3,)))


# Refused: the same, through two frames.
def forwarded_twice(x):
    return tf.matmul(x, x)


def call_forwarded_twice_inner(x):
    return forwarded_twice(x)


def call_forwarded_twice(x):
    return call_forwarded_twice_inner(x)


forwarded_twice(tf.ones((2, 2)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    call_forwarded_twice(tf.ones((3,)))


# Refused: one frame reached along two paths, since the frame between calls `call_diamond` from two
# sites.
def diamond(x):
    return tf.matmul(x, x)


def call_diamond(x):
    return diamond(x)


def call_call_diamond_twice(x):
    call_diamond(x)
    return call_diamond(x)


diamond(tf.ones((2, 2)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    call_call_diamond_twice(tf.ones((3,)))


# Converted: the frame between catches every exception and raises the declared one in its place,
# so the guard sees InvalidArgumentError whichever exception the call raises.
def caught(x):
    return tf.matmul(x, x)


def call_caught(x):
    try:
        return caught(x)
    except Exception:
        raise tf.errors.InvalidArgumentError(None, None, "rejected")


caught(tf.ones((2, 2)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    call_caught(tf.ones((3,)))
