from typing import Optional, Text, Tuple, Union

import numpy as np
import tensorflow as tf

a = np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]).astype(np.float32)
b = np.asarray([[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]]).astype(np.float32)


class Base(tf.keras.layers.Layer):
    def call(self, inputs):
        x0, x = inputs
        return tf.matmul(x0, x)


class WithBuild(tf.keras.layers.Layer):
    def build(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError("inputs type should be `tuple`.")
        self.built = True

    def call(self, inputs):
        x0, x = inputs
        return tf.matmul(x0, x)


class WithHint(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        x0, x = inputs
        return tf.matmul(x0, x)


class WithCtor(tf.keras.layers.Layer):
    def __init__(
        self,
        feature_map: Optional[int] = 3,
        activation: Union[Text, None] = "sigmoid",
        **kwargs,
    ):
        super(WithCtor, self).__init__(**kwargs)
        self._feature_map = feature_map
        self._activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        x0, x = inputs
        return self._activation(tf.matmul(x0, x))


class Symbolic(tf.keras.layers.Layer):
    def call(self, inputs):
        x0, x = inputs
        return tf.matmul(x0, x, transpose_b=True)


Base()((a, b))
WithBuild()((a, b))
WithHint()((a, b))
WithCtor(feature_map=2, activation="relu")((a, b))

# The symbolic pair and the chained call, as the reported subject has them.
Symbolic()((a, np.transpose(b, (0, 2, 1))))
inp = tf.keras.layers.Input(shape=(12, 10))
chained = Symbolic()((inp, inp))
