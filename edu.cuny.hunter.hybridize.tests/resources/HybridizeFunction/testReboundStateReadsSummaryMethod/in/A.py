# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1140

# Each function creates its own tape and looks up its gradient method. The engine's tf.GradientTape summary binds the method on each tape
# it creates, and tapes are told apart by allocation site, but no tape's method is rebound between two of its reads.

import tensorflow as tf

w = tf.Variable([1.0, 2.0])


def gradient_of_sum(x):
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(w * x)
    return tape.gradient(y, w)


def gradient_of_square(x):
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(w * w * x)
    return tape.gradient(y, w)


for i in range(2):
    print(gradient_of_sum(tf.fill([2], i + 1.0)))
    print(gradient_of_square(tf.fill([2], i + 1.0)))
