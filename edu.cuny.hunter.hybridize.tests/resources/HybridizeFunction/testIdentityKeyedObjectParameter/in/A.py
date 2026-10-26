# `tf.function` keys an argument that is an object of a program class by its identity, and a list, a tuple, a dict, a `namedtuple` or
# an `attrs` object by its contents (https://www.tensorflow.org/guide/function#rules_of_tracing). A primitive field of an object keyed by
# identity never causes a retrace, so it doesn't make a primitive parameter; a primitive element of a container does.
# - `boxed`: an object of a program class whose field is an `int` that differs across calls.
# - `listed`, `tupled`, `keyed`: a list, a tuple and a dict whose element is an `int` that differs across calls.
# - `boxes`: a list whose element is an object of a program class with an `int` field.
# - `setted`: a set whose `int` differs across calls, which `tf.function` can't key, raising a `TypeError`.
# - `enumerated`: an `enumerate` object over a list whose `int` differs across calls, which `tf.function` keys by its identity.
# - `named`: a `namedtuple` whose field is an `int` that differs across calls.
# - `attributed`: an `attrs` object whose field is an `int` that differs across calls.
import collections

import attr
import tensorflow as tf


class Box:
    def __init__(self, n):
        self.n = n


Pair = collections.namedtuple("Pair", ["n", "m"])


@attr.s
class Attributed:
    n = attr.ib()


def boxed(x, box):
    return x * box.n


def listed(x, ns):
    return x * ns[0]


def tupled(x, ns):
    return x * ns[0]


def keyed(x, ns):
    return x * ns["n"]


def boxes(x, bs):
    return x * bs[0].n


def setted(x, s):
    return x * 2


def enumerated(x, e):
    return x * 2


def named(x, p):
    return x * p.n


def attributed(x, a):
    return x * a.n


t = tf.ones([2])

boxed(t, Box(1))
boxed(t, Box(2))
listed(t, [1])
listed(t, [2])
tupled(t, (1,))
tupled(t, (2,))
keyed(t, {"n": 1})
keyed(t, {"n": 2})
boxes(t, [Box(1)])
boxes(t, [Box(2)])
setted(t, {1})
setted(t, {2})
enumerated(t, enumerate([1]))
enumerated(t, enumerate([2]))
named(t, Pair(1, 1))
named(t, Pair(2, 2))
attributed(t, Attributed(1))
attributed(t, Attributed(2))
