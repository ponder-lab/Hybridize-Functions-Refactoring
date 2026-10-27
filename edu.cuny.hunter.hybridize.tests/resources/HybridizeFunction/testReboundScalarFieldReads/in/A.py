# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1048
# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1076

# A Python scalar read from a field of a plain object. tf.function keys such an object by its identity, so a rewrite of the field never
# retraces, and a trace keeps the scalar the field held when it was traced.
# - `read_box`: reads `box.n`, which `set_box` rewrites between calls.
# - `Counter.read`: reads `self.n`, which `Counter.bump` rewrites between calls.
# - `read_fixed` and `Fixed.read`: read a field set only in `__init__`, which is read live.

import tensorflow as tf


class Box:
    def __init__(self, n):
        self.n = n


def read_box(x):
    return x * box.n


def set_box(n):
    box.n = n


class Counter:
    def __init__(self):
        self.n = 1

    def bump(self, n):
        self.n = n

    def read(self, x):
        return x * self.n


class Fixed:
    def __init__(self, n):
        self.n = n

    def read(self, x):
        return x * self.n


def read_fixed(x):
    return x * fixed.n


box = Box(1)
print(read_box(tf.constant([1.0, 2.0])))
set_box(5)
print(read_box(tf.constant([1.0, 2.0])))

counter = Counter()
print(counter.read(tf.constant([1.0, 2.0])))
counter.bump(5)
print(counter.read(tf.constant([1.0, 2.0])))

fixed = Fixed(7)
print(read_fixed(tf.constant([1.0, 2.0])))
print(read_fixed(tf.constant([1.0, 2.0])))
print(fixed.read(tf.constant([1.0, 2.0])))
print(fixed.read(tf.constant([1.0, 2.0])))
