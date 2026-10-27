# Fixture for a list a comprehension builds: it has three elements at run time, which the
# analysis does not position, so it gives the list no length and no nested specification is
# written.
import tensorflow as tf


def f(xs):
    return xs[0]


f([tf.ones([2]) for _ in range(3)])
