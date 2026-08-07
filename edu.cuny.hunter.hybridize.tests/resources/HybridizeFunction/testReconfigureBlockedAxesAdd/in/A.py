import tensorflow as tf


@tf.function
def wild(x):
    return tf.range(x.shape[1] - 1, -1, -1)


# Called at one rank with differing extents, so the inferred spec keeps rank 2 with axis 1
# dynamic while the body reads that axis statically into a weight-shape sink: the
# reconfiguration (adding the inferred signature) is otherwise viable and the unresolved
# axis is the sole blocker. Under per-call tracing the reads are concrete and both calls run.
wild(tf.ones((2, 3)))
wild(tf.ones((2, 5)))
