import tensorflow as tf


# Element 0 is dense at one call and sparse at the other, element 1 is float32 at one and int32 at the other, and element 2 differs in
# both, so each position fails, and the last one for two reasons.
def triple(xs):
    return xs[0], xs[1], xs[2]


triple([tf.ones([2]), tf.ones([2]), tf.ones([2])])
triple(
    [
        tf.sparse.from_dense(tf.ones([2])),
        tf.ones([2], dtype=tf.int32),
        tf.sparse.from_dense(tf.ones([2], dtype=tf.int32)),
    ]
)
