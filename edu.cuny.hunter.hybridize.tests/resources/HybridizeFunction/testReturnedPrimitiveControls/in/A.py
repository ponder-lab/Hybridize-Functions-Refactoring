# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1117

import tensorflow as tf


def scaled(x):
    return x * 2.0, {"norm": tf.norm(x)}


def accumulate(x):
    total = tf.reduce_sum(x)
    tf.debugging.assert_all_finite(total, "not finite")
    return None


def scaled_and_size(x):
    return x * 2.0, tf.size(x)


def discarded_width(x):
    tf.debugging.assert_all_finite(x, "not finite")
    return 6


t = tf.constant([1.0, 2.0])

y, stats = scaled(t)
print(y.numpy(), stats["norm"].numpy())

print(accumulate(t) is None)

y, n = scaled_and_size(t)
print(y.numpy(), int(n))

discarded_width(t)
discarded_width(t * 2.0)
