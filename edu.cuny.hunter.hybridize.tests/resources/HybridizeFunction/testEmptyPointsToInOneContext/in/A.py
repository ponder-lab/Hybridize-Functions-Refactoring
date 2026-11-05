# Fixture for a parameter that the analysis models in one calling context and not in another:
# `mixed` and `hybrid_mixed` are each passed an int literal and the result of string formatting,
# which the analysis doesn't model. What reaches them there is unknown, so neither is determined
# not to take a tensor.
import tensorflow as tf


def mixed(n):
    return n


@tf.function
def hybrid_mixed(n):
    return n


print(mixed(2))
print(mixed("_%d" % 3))
print(hybrid_mixed(2))
print(hybrid_mixed("_%d" % 3))
