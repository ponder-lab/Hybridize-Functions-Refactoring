import tensorflow as tf


# Receives a negated int32 tensor as `y`.
def select(x, y):
    return tf.where(x > 0, x, y)


select(tf.ones(3, dtype=tf.int32), -tf.ones(3, dtype=tf.int32))
