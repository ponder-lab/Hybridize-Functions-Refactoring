# Fixture for a parameter passed only a value the analysis doesn't model, the result of string
# formatting: nothing says whether `count`'s `n` is a tensor. `literal`'s `n` is passed int
# literals, which the analysis models.
import tensorflow as tf


def count(n):
    return len(n)


def literal(n):
    return n + 1


print(count("_%d" % 3))
print(literal(5))
print(literal(6))
