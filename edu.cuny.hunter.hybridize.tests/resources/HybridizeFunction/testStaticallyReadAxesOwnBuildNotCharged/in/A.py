import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.size_per_head = int(input_shape[-1] / 2)
        super().build(input_shape)

    def softmax(self, x, axis=-1):
        x = x - tf.reduce_max(x, axis=axis, keepdims=True)
        ex = tf.exp(x)
        return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

    def call(self, q):
        distance = tf.linalg.matmul(q, q, transpose_b=True)
        return self.softmax(distance)


Attention()(tf.random.uniform([2, 4, 3, 5]))
Attention()(tf.random.uniform([2, 4, 6, 7]))
