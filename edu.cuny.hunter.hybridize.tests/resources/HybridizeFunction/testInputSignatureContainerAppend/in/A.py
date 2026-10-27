# Fixture for a list that a single `append` grows after its display: the display has one
# element, but the list `f` receives has two. The analysis gives no length for a list an
# `append` reads off, so no nested specification is written; one of length 1 would reject
# the call.
import tensorflow as tf


def f(xs):
    return xs[0]


l = [tf.ones([2])]
l.append(tf.ones([2]))
f(l)
