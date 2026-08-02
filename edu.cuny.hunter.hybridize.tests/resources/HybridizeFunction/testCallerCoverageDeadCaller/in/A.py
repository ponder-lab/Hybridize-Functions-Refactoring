import tensorflow as tf


def covered(x):
    return tf.reduce_sum(x)


@tf.function
def live_caller(x):
    return covered(x) * 2.0


# Defined but never referenced: an unreachable eager call site contributes no executed
# path and no call-graph node, so it must not break `covered`'s coverage (the
# resize/load_image_test shape from the corpus evidence on #826).
def dead_caller(x):
    return covered(x) + 1.0


t = tf.ones((2, 2))
assert live_caller(t).numpy() == 8.0
