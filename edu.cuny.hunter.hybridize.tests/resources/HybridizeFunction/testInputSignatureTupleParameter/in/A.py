# Fixture for #781 over a TUPLE: `xs` receives a singleton tuple of one concrete tensor. Tuples
# are positionally-indexed sequences like lists, and TF 2.9.3 does not distinguish list from
# tuple in input_signature structure matching (probed: a list-shaped spec accepts a tuple
# argument and vice versa), so the parameter reduces to the same nested rendering a list would.
import tensorflow as tf


def f(xs):
    return xs[0]


f((tf.constant([1.0, 2.0]),))
