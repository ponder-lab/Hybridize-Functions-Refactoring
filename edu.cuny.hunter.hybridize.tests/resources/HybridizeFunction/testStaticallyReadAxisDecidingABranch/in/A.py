import tensorflow as tf


# The witness shape, reduced from a corpus layer: a statically-read axis decides a branch. Two
# call sites disagree on the last extent, so a signature generalizes that axis to a wildcard,
# and under a wildcard the read is `None`. The comparison then answers the same way whatever is
# passed, and the branch it decides silently computes the wrong thing. Comparisons were excluded
# from the sink set on the reasoning that a comparison under a wildcard is false rather than a
# raise: true of the comparison, false of the program, which either takes the wrong branch or,
# as in the corpus function, raises inside the one it takes.
def guarded(x):
    if x.shape[-1] != 4:
        return x * 3.0

    return x * 2.0


# The same comparison, with its result returned instead of deciding anything. A changed result
# changes this value and no control flow, so it is not a sink; flagging it would decline a
# function that is fine.
def compared_only(x):
    return x.shape[-1] == 4


# A dynamic read deciding a branch. `tf.shape` yields a tensor rather than a Python integer, so
# it is correct under any wildcard and must not be flagged.
def dynamic_guard(x):
    return tf.cond(tf.equal(tf.shape(x)[-1], 4), lambda: x * 2.0, lambda: x * 3.0)


a = tf.ones((2, 4))
b = tf.ones((2, 5))

assert guarded(a).shape == (2, 4)
assert guarded(b).shape == (2, 5)
assert compared_only(a)
assert not compared_only(b)
assert dynamic_guard(a).shape == (2, 4)
assert dynamic_guard(b).shape == (2, 5)
