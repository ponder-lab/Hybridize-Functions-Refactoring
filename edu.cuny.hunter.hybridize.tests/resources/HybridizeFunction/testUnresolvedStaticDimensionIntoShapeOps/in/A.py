# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1058

import tensorflow as tf


def wild_ones(x):
    return tf.ones((x.shape[0], 1)) + tf.reduce_sum(x)


def wild_ones_keyword(x):
    return tf.ones(shape=(x.shape[0], 1)) + tf.reduce_sum(x)


def wild_zeros(x):
    return tf.zeros((x.shape[0], 1)) + tf.reduce_sum(x)


def wild_fill(x):
    return tf.fill((x.shape[0], 1), 1.0) + tf.reduce_sum(x)


def wild_fill_value(x):
    return tf.fill((2, 1), x.shape[0]) + tf.cast(tf.reduce_sum(x), tf.int32)


def wild_eye(x):
    return tf.eye(x.shape[0]) + tf.reduce_sum(x)


def wild_normal(x):
    return tf.random.normal((x.shape[0], 1)) + tf.reduce_sum(x)


def wild_uniform(x):
    return tf.random.uniform((x.shape[0], 1)) + tf.reduce_sum(x)


def wild_truncated_normal(x):
    return tf.random.truncated_normal((x.shape[0], 1)) + tf.reduce_sum(x)


def wild_broadcast_to(x):
    return tf.broadcast_to(tf.reduce_sum(x), (x.shape[0], 1))


def wild_tile(x):
    return tf.tile(tf.reshape(tf.reduce_sum(x), (1, 1)), (x.shape[0], 1))


def wild_one_hot(x):
    return tf.one_hot(tf.constant([0]), x.shape[0]) + tf.reduce_sum(x)


def wild_split(x):
    return tf.split(tf.ones((4,)) * tf.reduce_sum(x), x.shape[0])


def wild_range(x):
    acc = tf.reduce_sum(x)
    for _ in range(x.shape[0]):
        acc = acc + 1.0
    return acc


def wild_int(x):
    n = int(x.shape[0])
    return tf.reduce_sum(x) + tf.constant(n, tf.float32)


# The dynamic shape is a tensor, valid under a wildcard axis.
def dynamic_ones(x):
    return tf.ones((tf.shape(x)[0], 1)) + tf.reduce_sum(x)


# Every call passes the same extent, so the signature resolves the axis the body reads.
def ranked_ones(x):
    return tf.ones((x.shape[0], 1)) + tf.reduce_sum(x)


for function in (
    wild_ones,
    wild_ones_keyword,
    wild_zeros,
    wild_fill,
    wild_fill_value,
    wild_eye,
    wild_normal,
    wild_uniform,
    wild_truncated_normal,
    wild_broadcast_to,
    wild_tile,
    wild_one_hot,
    wild_split,
    wild_range,
    wild_int,
    dynamic_ones,
):
    function(tf.ones([2, 3]))
    function(tf.ones([4, 3]))

ranked_ones(tf.ones([2, 3]))
ranked_ones(tf.ones([2, 3]))
