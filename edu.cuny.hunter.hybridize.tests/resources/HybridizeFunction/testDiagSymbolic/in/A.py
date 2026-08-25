from typing import Tuple

import numpy as np
import tensorflow as tf

a = np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]).astype(np.float32)
b = np.asarray([[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]]).astype(np.float32)


class SymbolicOnly(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x0, x = inputs
        return tf.matmul(x0, x, transpose_b=True)


class Chained(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x0, x = inputs
        return x0 * tf.reduce_mean(x)


class EagerOnly(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x0, x = inputs
        return tf.matmul(x0, x)


# Symbolic construction only: a pair of `Input` results, whose shape and dtype are known.
i0 = tf.keras.layers.Input(shape=(12, 10))
SymbolicOnly()((i0, i0))

# The chained shape: the second call takes the previous layer's output.
first = Chained()((i0, i0))
Chained()((i0, first))

# Eager only, as the control.
EagerOnly()((a, b))


# Both kinds on one parameter, which is the shape the reported subject has.
class Mixed(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x0, x = inputs
        return x0 * tf.reduce_mean(x)


Mixed()((a, a))
Mixed()((i0, i0))


# The same mixture, except one element is produced by `np.transpose` rather than a literal.
class Transposed(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x0, x = inputs
        return x0 * tf.reduce_mean(x)


Transposed()((a, np.transpose(b, (0, 2, 1))))
Transposed()((i0, i0))
