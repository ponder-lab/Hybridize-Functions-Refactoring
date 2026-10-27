import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec([2, 3], tf.float32)])
def wild(x):
    return tf.range(x.shape[0] - 1, -1, -1)


# The inferred wildcard comes from flow over-approximation, not runtime variation: the
# zero-trip loop's body never executes, so only (2, 3) reaches the call, conforming to the
# supplied signature with the trace-time read of axis 0 a concrete 2, and the program runs.
# The analysis's phi-merge at the loop head joins both definitions regardless of the trip
# count, so the parameter types as both shapes and the inferred spec joins axis 0 to a
# wildcard the body reads statically: the otherwise-viable reconfiguration is blocked by the
# axis alone (#865's supplied-signature arm). Straight-line constructions cannot produce
# this: the analysis constant-folds reshape's -1, enumerates a dataset's batch shapes
# including the remainder batch, and prunes infeasible constant branches, so a wild inferred
# axis over a uniform runtime needs a merge point the analysis keeps, and the loop-head phi
# is exactly that.
b = tf.ones((2, 3))
for _ in range(0):
    b = tf.ones((5, 3))

wild(b)
