# https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/1140

# The front end lowers each super() call to a proxy object that the calling method allocates and initializes. A function reaching a
# super() call reads the proxy's fields, which are no program state.

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self._units = units
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self._units), initializer="ones", name="w"
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attentions = [Attention(2) for _ in range(2)]
        super(Encoder, self).build(input_shape)

    def encoder(self, inputs):
        encodings = inputs
        for i in range(2):
            attention = self.attentions[i]
            encodings = attention(encodings)
        return encodings

    def call(self, inputs):
        return self.encoder(inputs)


model = Encoder()
print(model(tf.ones([1, 2])))
print(model(tf.fill([1, 2], 2.0)))
