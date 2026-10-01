import tensorflow as tf

# Pins #1014 through a Keras layer: `LanguageModel.call` is reached from the test through the
# layer's `__call__`, and the call declared to fail passes a dict whose element fails `tf.matmul`'s
# static shape check, which a bare decorator raises as ValueError at trace time in place of the
# InvalidArgumentError the guard declares.


class Model(tf.keras.layers.Layer):
    def __call__(self, features, labels=None, training=None, step=None):
        outputs, predictions = super().__call__(
            features, labels=labels, training=training, step=step
        )
        return outputs, predictions


class LanguageModel(Model):
    def call(self, features, labels=None, training=None, step=None):
        y = tf.matmul(features["a"], features["a"])
        return y, y


class ModelTest(tf.test.TestCase):
    def testLanguageModel(self):
        model = LanguageModel()
        model({"a": tf.ones((2, 2))})

    def testLanguageModelWithMissingStart(self):
        model = LanguageModel()
        with self.assertRaises(tf.errors.InvalidArgumentError):
            model({"a": tf.ones((3,))})
