# Fixture for #976's #892 boundary: `x` receives a bare tensor at its conforming call site and a
# list of tensors only at a call the tests declare must fail. The container verdict over every
# node is TRUE, but no conforming caller passes a container, so the flat reduction of the
# conforming evidence stands and the signature is kept. Blocking it as an unmodeled container
# would report the form unsupported on the strength of a rejected call. The conforming call
# precedes the guard deliberately, since a guard dominates every later call in the same frame.
import unittest

import tensorflow as tf

case = unittest.TestCase()


def scale(x):
    return x * 2.0


assert scale(tf.ones([2, 3])).shape == (2, 3)

with case.assertRaises(TypeError):
    scale([tf.ones([2]), tf.ones([3])])
