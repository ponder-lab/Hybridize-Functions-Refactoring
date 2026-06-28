import tensorflow as tf


# Regression guard for #429. The shape `[2 * 14]` is a literal-arithmetic expression that
# folds to the numeric dimension 28 only when Jython's interpreter is healthy under
# Tycho-OSGi (the frozen_importlib fragment puts the resource on the wrapped Ariadne
# bundle's classloader). On a degraded interpreter the constant-folding pass falls back to
# a symbolic dimension, so `t`'s shape would be symbolic rather than 28.
def func(t):
    return tf.reduce_sum(t)


func(tf.zeros([2 * 14]))
