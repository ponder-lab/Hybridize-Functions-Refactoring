import tensorflow as tf


class Lazy(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 2))
        super().build(input_shape)

    def call(self, x):
        if not self.built:
            self.build(x.shape)
        return tf.linalg.matmul(x, self.w)


Lazy().call(tf.random.uniform([2, 3]))
Lazy().call(tf.random.uniform([4, 5]))
