import numpy as np
import tensorflow as tf


def build_pair():
    # The dual channel. Each name is initialized to None and only then conditionally reassigned,
    # so ONE field of the returned list carries both the null and the array's tensor state. Two
    # elements in different positions do not produce this: one position would hold only the null
    # and the other only the array, and both of those are already handled.
    #
    # The guard is data-dependent so nothing can prune the branch that is dead at run time.
    attributes = None
    labels = None
    names = ["labels"]

    if "attributes" in names:
        attributes = np.array([[1.0, 2.0], [3.0, 4.0]])

    if "labels" in names:
        labels = np.array([[5.0], [6.0]])

    return [attributes, labels]


def concatenate_present(seq):
    # `seq`'s leading element is None on every reachable path, and no `tf.TensorSpec` admits None.
    # A signature covers every parameter or none, so the honest emission here is an absence rather
    # than a specification whose leading position cannot match the call it was derived from.
    present = [x for x in seq if x is not None]
    return tf.reduce_sum(tf.cast(present[0], tf.float32))


def build_pair_without_none():
    # The sibling control: same shape, both elements allocated unconditionally, no None anywhere.
    # Its parameter must keep its signature after the fix. A fix asserted only against the case
    # above cannot show it left this one alone.
    return [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0], [6.0]])]


def concatenate_all(seq):
    return tf.reduce_sum(tf.cast(seq[0], tf.float32)) + tf.reduce_sum(
        tf.cast(seq[1], tf.float32)
    )


concatenate_present(build_pair())
concatenate_all(build_pair_without_none())
