# Fixture for #781's arity bottom: `xs` receives a singleton list at one call site and a
# two-element list at the other. TensorFlow rejects a sequence of a different length than the
# signature declares (ValueError on 2.9.3) and no wildcard length exists, so no single nested
# spec admits both call sites and the reduction reports HETEROGENEOUS_ARITY, the same
# |X| != 1 => bottom discipline the dtype and sparseness axes use.
import tensorflow as tf


def f(xs):
    return xs[0]


t = tf.constant([1.0, 2.0])
f([t])
f([t, t])
