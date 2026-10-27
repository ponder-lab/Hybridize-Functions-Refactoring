# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1140

# An opaque object passed as an argument, whose field another function rebinds. tf.function keys such an argument by identity, so a trace
# keeps the tensor the field held when it was traced.

import tensorflow as tf


class Weights:
    def __init__(self):
        self.w = tf.constant([1.0, 1.0])

    def set_weights(self, w):
        self.w = w


def apply(weights, x):
    return x * weights.w


# A class attribute, set in the class body and rebound by the module, which allocates the class object.
class Scale:
    factor = tf.constant(2.0)


def scaled(x):
    return x * Scale.factor


for i in range(2):
    Scale.factor = tf.constant(i + 2.0)
    print(scaled(tf.constant([1.0, 2.0])))

weights = Weights()
for i in range(2):
    weights.set_weights(tf.fill([2], i + 2.0))
    print(apply(weights, tf.constant([1.0, 2.0])))
