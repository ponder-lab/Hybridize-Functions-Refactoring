# Fixture for the comprehension-built sequence reduction (#807): the container is
# built by a list comprehension, whose catalog at Ariadne 0.52.58+ carries the
# analysis-internal append-contents key alongside the numeric indices. With the
# synthetic key filtered, the contiguous singleton reduces exactly like the
# literal-list fixture.
import tensorflow as tf


def f(xs):
    return xs[0]


f([tf.constant([1.0, 2.0]) for _ in range(1)])
