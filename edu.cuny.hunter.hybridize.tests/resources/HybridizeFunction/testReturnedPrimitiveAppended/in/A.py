# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1117: a primitive returned as an element of a list built by
# `append`, a literal in `appended` and a computed `int` in `appended_count`, or built by concatenation, in `concatenated`, and
# repetition, in `repeated`.

import tensorflow as tf


def appended(x):
    out = []
    out.append(x * 2.0)
    out.append(3)
    return out


def appended_count(x, xs):
    out = []
    out.append(x * 2.0)
    out.append(len(xs))
    return out


def concatenated(x):
    return [x * 2.0] + [3]


def repeated(x):
    return [x * 2.0] + [3] * 2


t = tf.constant([1.0, 2.0])

r = appended(t)
print({r[1]: "appended"}[3])

r = appended_count(t, [t, t])
print({r[1]: "appended_count"}[2])

r = concatenated(t)
print({r[1]: "concatenated"}[3])

r = repeated(t)
print({r[2]: "repeated"}[3])
