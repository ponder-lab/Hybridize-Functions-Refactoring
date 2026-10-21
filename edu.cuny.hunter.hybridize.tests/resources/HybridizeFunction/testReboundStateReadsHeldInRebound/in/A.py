# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1140

# A NamedTuple or a tf.concat result held in a field another method rebinds. A trace keeps the object the field held when it was traced.

from typing import NamedTuple

import tensorflow as tf


class G(NamedTuple):
    e: tf.Tensor


class Holder:
    def __init__(self):
        self.g = G(tf.constant([1.0]))

    def reset(self, v):
        self.g = G(v)

    def read(self, x):
        return x * self.g.e


class Box:
    def __init__(self):
        self.t = tf.concat([tf.zeros([1, 2]), tf.zeros([1, 2])], 0)

    def reset(self, v):
        self.t = tf.concat([v, v], 0)

    def read(self, x):
        total = tf.zeros([2])
        for row in self.t:
            total += row
        return x + total


holder = Holder()
box = Box()
for i in range(2):
    holder.reset(tf.constant([i + 2.0]))
    print(holder.read(tf.constant([1.0])))
    box.reset(tf.fill([1, 2], i + 2.0))
    print(box.read(tf.constant([1.0, 1.0])))
