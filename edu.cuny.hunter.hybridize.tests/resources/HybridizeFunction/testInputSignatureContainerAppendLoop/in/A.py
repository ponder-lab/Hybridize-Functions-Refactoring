# Fixture for #781's designed unsupported form: `xs` receives a list grown by an append loop, so
# its object catalog is not a contiguous run of constant indices the extraction can enumerate.
# The parameter still classifies as a tensor container (Phase 3), but no element structure is
# extractable, and the signature drops with TENSOR_CONTAINER_UNSUPPORTED. This is the
# GraphSAGE-style shape in the corpus (messages accumulated per edge type).
import tensorflow as tf


def f(xs):
    return xs[0]


l = []
for i in range(3):
    l.append(tf.ones([2]))
f(l)
