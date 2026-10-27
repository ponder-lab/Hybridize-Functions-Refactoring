# Issues 1025 and 1109, control. The primitive `n` is 0 at the first call and then each index of a `range` loop, so it takes three
# values and traces three times under `tf.function`. It must still count as a primitive parameter.
import tensorflow as tf


def f(x, n):
    return x * n


f(tf.ones([2]), 0)

for i in range(3):
    f(tf.ones([2]), i)
