import tensorflow as tf


# A tf.keras.Model whose call returns a tuple, invoked via the Keras __call__ (`self(inputs)`), with
# the result tuple-unpacked, then passed to `loss`. `pred` should be tensor-typed.
class M(tf.keras.Model):
    def call(self, x, training=True):
        return x * 2.0, None

    def loss(self, real, pred):
        return tf.reduce_mean(tf.square(pred - real))

    def step(self, inputs, targets):
        predictions, _ = self(inputs, training=True)
        return self.loss(targets, predictions)


M().step(tf.constant([1.0, 2.0]), tf.constant([1.0, 1.0]))
