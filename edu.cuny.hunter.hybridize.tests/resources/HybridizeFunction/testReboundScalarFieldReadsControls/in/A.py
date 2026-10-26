# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1048
# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1076

# Scalar reads that no other function rewrites, and the scalars the engine computes or doesn't.
# - `module_scaled`: reads `config.N`, a module constant set only where the module defines it.
# - `Rated.self_scaled` and `Rated.class_scaled`: read the class-body constant `RATE` as `self.RATE` and as `Rated.RATE`.
# - `switched`: takes `training=True` as a keyword at two call sites.
# - `Grown.read`: reads `self.n`, only ever a computed number, which `Grown.grow` rewrites between calls.
# - `Prefixed.read`: reads `self.prefix`, only ever a computed string, which `Prefixed.rename` rewrites between calls.
# - `Retuned.scaled`: reads the class-body constant `Retuned.RATE`, which the module reassigns to another literal between calls.

import config
import tensorflow as tf


def module_scaled(x):
    return x * config.N


class Rated:
    RATE = 0.1

    def self_scaled(self, x):
        return x * self.RATE

    def class_scaled(self, x):
        return x * Rated.RATE


def switched(x, training=False):
    if training:
        return x * 2.0
    return x


class Grown:
    def __init__(self, t):
        self.n = t.shape[0] * t.shape[1]

    def grow(self):
        self.n = self.n * 2

    def read(self, x):
        return x * self.n


class Prefixed:
    def __init__(self, i):
        self.prefix = "p" + str(i)

    def rename(self, i):
        self.prefix = "q" + str(i)

    def read(self, x):
        return tf.strings.join([self.prefix, x])


class Retuned:
    RATE = 0.1

    def scaled(self, x):
        return x * Retuned.RATE


t = tf.constant([1.0, 2.0])

print(module_scaled(t))
print(module_scaled(t))

rated = Rated()
print(rated.self_scaled(t))
print(rated.class_scaled(t))

print(switched(t, training=True))
print(switched(t, training=True))

grown = Grown(tf.ones([2, 3]))
print(grown.read(t))
grown.grow()
print(grown.read(t))

prefixed = Prefixed(1)
print(prefixed.read(tf.constant("x")))
prefixed.rename(2)
print(prefixed.read(tf.constant("x")))

retuned = Retuned()
print(retuned.scaled(t))
Retuned.RATE = 0.2
print(retuned.scaled(t))
