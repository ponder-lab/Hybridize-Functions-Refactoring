import numpy as np
import tensorflow as tf


def plain_tf(inputs):
    x0, x = inputs
    return tf.matmul(x0, x)


def plain_np(inputs):
    x0, x = inputs
    return tf.matmul(x0, x)


class LayerTf(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        x0, x = inputs
        return tf.matmul(x0, x)


@tf.keras.utils.register_keras_serializable()
class LayerNp(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        x0, x = inputs
        return tf.matmul(x0, x)


class LayerInMethod(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        x0, x = inputs
        return tf.matmul(x0, x)


class Harness(tf.test.TestCase):
    def test_call(self):
        x0 = np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]).astype(np.float32)
        x = np.asarray([[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]]).astype(np.float32)
        LayerInMethod()((x0, x))


plain_tf((tf.ones((2, 3, 5)), tf.ones((2, 5, 3))))

a = np.asarray([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]).astype(np.float32)
b = np.asarray([[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]]).astype(np.float32)
plain_np((a, b))

LayerTf()((tf.ones((2, 3, 5)), tf.ones((2, 5, 3))))
LayerNp()((a, b))

Harness().test_call()
