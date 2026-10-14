# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1058

import tensorflow as tf

# Each of these arguments accepts None and silently computes something else: a dimension read statically from a wild axis is None at
# trace time, so the traced function disagrees with the eager one without raising.


def meaning_one_hot_on_value(x):
    hot = tf.one_hot(tf.constant([0]), 3, on_value=x.shape[0])
    return tf.cast(hot, tf.float32) + tf.reduce_sum(x)


def meaning_one_hot_off_value(x):
    hot = tf.one_hot(tf.constant([0]), 3, off_value=x.shape[0])
    return tf.cast(hot, tf.float32) + tf.reduce_sum(x)


def meaning_normal_seed(x):
    return tf.random.normal((2, 1), seed=x.shape[0]) + tf.reduce_sum(x)


def meaning_uniform_seed(x):
    return tf.random.uniform((2, 1), seed=x.shape[0]) + tf.reduce_sum(x)


def meaning_sequence_mask(x):
    mask = tf.sequence_mask([1, 2], maxlen=x.shape[0])
    return tf.cast(mask, tf.float32) + tf.reduce_sum(x)


# With an integer `num_or_size_splits`, `num` is unused, so its None changes nothing; it is counted anyway, as the simpler rule.
def counted_split_num(x):
    return tf.split(tf.ones((4,)) * tf.reduce_sum(x), 2, num=x.shape[0])


for function in (
    meaning_one_hot_on_value,
    meaning_one_hot_off_value,
    meaning_normal_seed,
    meaning_uniform_seed,
    meaning_sequence_mask,
    counted_split_num,
):
    function(tf.ones([2, 3]))
    function(tf.ones([4, 3]))
