# Issues 1025 and 1109, control. Each primitive has one constant in its points-to set, but some call passes a value the analysis
# has no abstract value for (a `range` loop index), so the set isn't the complete set of values, and the primitive still counts.
# - `scaled`: the call `scaled(x, i)` has its own calling context, in which the parameter points to nothing.
# - `relayed`: reached from one call site in `relay`, whose two calling contexts pass 0.5 and `i`; the parameter points to 0.5 alone,
#   and the argument at the call site points to nothing in the second context.
# - `defaulted`: its default comes from string formatting, which the analysis doesn't model, so in the calling context of the call
#   that omits it, the parameter points to nothing, though no call site passes it an argument without a value.
# - `listed`: a list is mutable, so its contents aren't a fixed value, even when each element is a single constant.
import tensorflow as tf


@tf.function
def scaled(x, n):
    return x * n


@tf.function
def relayed(x, n):
    return x * n


def relay(x, n):
    return relayed(x, n)


@tf.function
def defaulted(x, n=float("%d" % 3)):
    return x * n


@tf.function
def listed(x, means):
    return x - tf.constant(means, dtype=tf.float32)


scaled(tf.ones([2]), 0.5)
relay(tf.ones([2]), 0.5)

for i in range(3):
    scaled(tf.ones([2]), i)
    relay(tf.ones([2]), i)

defaulted(tf.ones([2]))
defaulted(tf.ones([2]), 0.5)

listed(tf.ones([2]), [0.0, 0.1])

print(scaled.experimental_get_tracing_count())  # 4
