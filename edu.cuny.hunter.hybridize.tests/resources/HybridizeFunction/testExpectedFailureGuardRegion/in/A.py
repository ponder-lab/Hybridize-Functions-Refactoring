import unittest

import tensorflow as tf

# Pins the guard region of #898: a guard covers its own `with` body, not everything written
# after it. `after_guard` is called once inside the block, where the rank-1 argument raises
# and the guard both expects and swallows the error, and once below the block, where a
# square argument multiplies cleanly. Only the first is a declared failure, so only its
# evidence is set aside and the surviving specification is the square one. Reading the guard
# as dominance alone put both calls inside it, which left every node of the function guarded
# and so excluded nothing at all.

case = unittest.TestCase()


def after_guard(x):
    return tf.matmul(x, x)


with case.assertRaises(tf.errors.InvalidArgumentError):
    after_guard(tf.ones((3,)))

assert after_guard(tf.ones((2, 2))).shape == (2, 2)
