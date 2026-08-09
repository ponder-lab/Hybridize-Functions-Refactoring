import unittest

import tensorflow as tf

# Reduces deep_recommenders' CIN.call (#888): `cin` receives a two-tensor tuple at its
# conforming call site and a bare tensor at a site the tests declare must fail. Ariadne
# types the parameter from the unwrapped site, and a single `TensorSpec` reduced from that
# site admits none of the callers that pass the tuple, so the specification must not be
# derived from it. The guarded call is legal Python that runs, since `assertRaises` both
# expects and swallows the error, which keeps the file executable while still declaring
# that the call fails. `tuples_only` receives tuples alone and keeps its nested spec.

case = unittest.TestCase()


def cin(inputs):
    return tf.matmul(inputs[0], inputs[1])


def tuples_only(inputs):
    return tf.matmul(inputs[0], inputs[1])


assert cin((tf.ones((2, 3, 5)), tf.ones((2, 5, 3)))).shape == (2, 3, 3)

with case.assertRaises(tf.errors.InvalidArgumentError):
    cin(tf.ones((2, 3, 5)))

assert tuples_only((tf.ones((2, 3, 5)), tf.ones((2, 5, 3)))).shape == (2, 3, 3)
