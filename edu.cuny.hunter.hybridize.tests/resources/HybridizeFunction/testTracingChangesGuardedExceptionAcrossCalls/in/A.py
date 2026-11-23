import unittest

import tensorflow as tf

# Pins #1321: two calls declared to fail with a TensorFlow error raise different exceptions once
# the function is traced. Eagerly, each raises InvalidArgumentError; under a bare decorator, the
# rank-1 argument fails `tf.matmul`'s static shape check with ValueError and the int32 argument
# fails `tf.sqrt`'s static dtype check with TypeError, on the pinned TensorFlow 2.9.3. The
# conforming call precedes the guards.

case = unittest.TestCase()


def two_guards(x):
    return tf.sqrt(tf.matmul(x, x))


two_guards(tf.ones((2, 2)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    two_guards(tf.ones((3,)))

with case.assertRaises(tf.errors.InvalidArgumentError):
    two_guards(tf.ones((2, 2), tf.int32))
