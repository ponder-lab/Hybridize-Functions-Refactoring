# Fixture for the comprehension-built sequence (#807): the container is built by a list
# comprehension, whose elements the analysis does not position. Its length is the iteration
# count, which the analysis does not know, so it gives the list no length and the parameter
# is not reduced to a nested specification, although this comprehension happens to run once.
import tensorflow as tf


def f(xs):
    return xs[0]


f([tf.constant([1.0, 2.0]) for _ in range(1)])
