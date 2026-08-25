from typing import Tuple

import numpy as np
import tensorflow as tf

# Pins the annotation half of #899. A type hint answers whether a parameter is tensor-like,
# not what specification may be written for it, and it carries no dtype of its own. Before,
# classification returned on the hint and the container question was never asked, which is
# what kept an annotated parameter out of reach of the recovery.
#
# `hinted` takes arrays, whose shapes do not survive, and `symbolic` takes a pair of `Input`
# results, whose shapes do. `unmodeled` takes an element produced by a call the analysis does
# not model, so no element evidence exists for that position and the form is unsupported.

a = np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]).astype(np.float32)
b = np.asarray([[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]]).astype(np.float32)


class Hinted(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x0, x = inputs
        return tf.matmul(x0, x)


class Symbolic(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x0, x = inputs
        return x0 * tf.reduce_mean(x)


class Unmodeled(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x0, x = inputs
        return x0 * tf.reduce_mean(x)


Hinted()((a, b))

i0 = tf.keras.layers.Input(shape=(12, 10))
Symbolic()((i0, i0))

Unmodeled()((a, np.transpose(b, (0, 2, 1))))
