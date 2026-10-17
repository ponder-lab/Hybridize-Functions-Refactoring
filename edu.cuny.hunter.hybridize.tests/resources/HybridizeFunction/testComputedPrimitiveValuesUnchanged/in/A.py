# Issues 1025 and 1109, control. The analysis gives a computed Python scalar (the result of a binary operation) no abstract value
# (wala/ML#805, wala/ML#398), so a value set merged with one keeps only its constant operands, and a non-empty points-to set isn't the
# complete set of values. Each primitive here takes six values, so each function traces six times under `tf.function`, and the
# primitive must still count.
# - `f`: `n` starts at 0 and is incremented in a loop; its points-to set holds only the constant 0.
# - `g`: the tuple's second element is computed, so the tuple's field for it points to nothing.
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
