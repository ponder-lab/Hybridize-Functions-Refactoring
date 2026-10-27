# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1074

import tensorflow as tf


def none_return(x):
    if tf.reduce_sum(x) > 0.0:
        return None
    return x + 1.0


def none_merge(x):
    y = None
    if tf.reduce_sum(x) > 0.0:
        y = x + 1.0
    return y


def float_of(x):
    return x * float(tf.reduce_sum(x))


def bool_of(x):
    if bool(tf.reduce_sum(x) > 0.0):
        return x + 1.0
    return x - 1.0


def via_callee(x):
    r = none_return(x)
    return x if r is None else r


def tensor_branches(x):
    if tf.reduce_sum(x) > 0.0:
        return x + 1.0
    return x - 1.0


def int_of(x):
    return tf.zeros((2,)) + tf.cast(int(tf.reduce_sum(x)), tf.float32)


def local_in_branch(x):
    if tf.reduce_sum(x) > 0.0:
        tmp = x + 1.0
        x = tmp * 2.0
    return x


def local_after_join(x):
    if tf.reduce_sum(x) > 0.0:
        tmp = x + 1.0
    return tmp


def primitive_branch(x, n):
    if n > 0:
        return None
    return x + float(n)


def static_none(x):
    if x.shape[0] == 3:
        return x + 1.0
    return None


def mask_or_none(inputs, sequence_length=None):
    if sequence_length is None:
        return None
    return tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1])


def dataset_size(dataset):
    if dataset.cardinality() == tf.data.INFINITE_CARDINALITY:
        return None
    return dataset.reduce(tf.constant(0, dtype=tf.int64), lambda count, _: count + 1)


def none_return_greater(x):
    if tf.greater(tf.reduce_sum(x), 0.0):
        return None
    return x + 1.0


def none_merge_greater(x):
    y = None
    if tf.greater(tf.reduce_sum(x), 0.0):
        y = x + 1.0
    return y


def local_after_join_greater(x):
    if tf.greater(tf.reduce_sum(x), 0.0):
        tmp = x + 1.0
    return tmp


def bool_of_greater(x):
    if bool(tf.greater(tf.reduce_sum(x), 0.0)):
        return x + 1.0
    return x - 1.0


def via_callee_greater(x):
    r = none_return_greater(x)
    return x if r is None else r


def tensor_branches_greater(x):
    if tf.greater(tf.reduce_sum(x), 0.0):
        return x + 1.0
    return x - 1.0


def none_return_unmodeled(x):
    if tf.reduce_any(tf.math.is_nan(x)):
        return None
    return x + 1.0


t = tf.constant([1.0, -2.0])
for f in (
    none_return,
    none_merge,
    float_of,
    bool_of,
    via_callee,
    tensor_branches,
    int_of,
):
    f(t)
static_none(tf.constant([1.0, 2.0, 3.0]))
local_in_branch(t)
primitive_branch(t, 0)
primitive_branch(t, 1)
local_after_join(tf.constant([1.0, 2.0]))
mask_or_none(tf.ones([2, 3]))
mask_or_none(tf.ones([2, 3]), tf.constant([1, 2]))
dataset_size(tf.data.Dataset.range(5))
none_return_greater(t)
bool_of_greater(t)
via_callee_greater(t)
tensor_branches_greater(t)
none_merge_greater(t)
local_after_join_greater(tf.constant([1.0, 2.0]))
none_return_unmodeled(t)
