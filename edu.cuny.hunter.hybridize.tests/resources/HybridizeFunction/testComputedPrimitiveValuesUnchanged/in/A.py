# Issues 1025 and 1109, control. A computed Python scalar (the result of a binary operation) points to its type's key (wala/ML#1149),
# not to a constant, so a value set merged with one isn't a single constant. Each primitive here takes six values, so each function
# traces six times under `tf.function`, and the primitive must still count.
# - `f`: `n` starts at 0 and is incremented in a loop; its computed values point to the `int` key, so it isn't the constant 0 alone.
# - `g`: the tuple's second element is computed, and the tuple isn't a constant.
import tensorflow as tf


def f(x, n):
    return x * n


def g(x, pair):
    return x * pair[1]


n = 0

while n < 6:
    f(tf.ones([2]), n)
    n += 1

for k in range(6):
    g(tf.ones([2]), (1, k * 2))
