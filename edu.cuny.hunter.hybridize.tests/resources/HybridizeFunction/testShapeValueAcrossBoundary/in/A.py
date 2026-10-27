import tensorflow as tf


class Projection(tf.keras.layers.Layer):
    def build(self, input_shape):
        # A lazy build reads the static last dimension of its input.
        self.kernel = self.add_weight(shape=[input_shape[-1], 3])

    def call(self, x):
        return tf.matmul(x, self.kernel)


class Gating(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.projection = Projection()

    def _pre_combine(self, x):
        x_shape = tf.shape(x)
        lost_shape = x_shape[1:]
        x = tf.reshape(x, [x_shape[0], tf.reduce_prod(lost_shape)])
        return x, lost_shape

    def _post_combine(self, x, lost_shape):
        batch_size = tf.expand_dims(tf.shape(x)[0], axis=0)
        return tf.reshape(x, tf.concat((batch_size, lost_shape), 0))

    def call(self, x):
        flat, lost_shape = self._pre_combine(x)
        flat = flat * 2.0
        restored = self._post_combine(flat, lost_shape)
        return self.projection(restored)


gating = Gating()
model = tf.keras.Sequential([gating])
# Keras traces the layer's call here, so a converted helper's boundary drops the shape's value.
print(model.predict(tf.ones([2, 4, 5])).shape)
# Called directly, after the trace, for the call graph to reach the layer's call.
print(gating(tf.ones([2, 4, 5])).shape)
