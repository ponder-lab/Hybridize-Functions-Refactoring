# Fixture for #781's per-element bottom: `xs` receives a singleton list at both call sites, but
# the element is float32 at one and int32 at the other. The container form is modeled (arity 1
# at both sites), so the reduction reaches the element position, where the dtype union is
# heterogeneous and reduces to bottom exactly as a flat parameter's would; the parameter blocks
# with HETEROGENEOUS_DTYPE and the diagnostic cites the element position.
import tensorflow as tf


def f(xs):
    return xs[0]


f([tf.ones([2])])
f([tf.constant([1, 2])])
