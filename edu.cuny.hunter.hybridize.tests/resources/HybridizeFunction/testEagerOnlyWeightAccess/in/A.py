import tensorflow as tf

reference = tf.keras.layers.Dense(2)
reference.build((None, 2))
SAVED = reference.get_weights()


class Reader(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.E = tf.keras.layers.Dense(2)
        self.E.build((None, 2))

    # The receiver is an ATTRIBUTE, not a local, which is the shape that matters. Its tensor
    # typing is unavailable, so the unconditional name match is the only thing that can catch
    # it; a reduced `get_weights()` on a local would pass while leaving that unexercised.
    def reads(self, x):
        return tf.matmul(x, tf.constant(self.E.get_weights()[0]))


class Writer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.E = tf.keras.layers.Dense(2)
        self.E.build((None, 2))

    # `set_weights` is NOT eager-only and must not block: `batch_set_value` builds assign ops
    # under tracing, while `batch_get_value` must extract concrete values and cannot. Verified
    # by running both inside a `tf.function` on the pinned TensorFlow.
    def writes(self, x):
        self.E.set_weights(SAVED)
        return self.E(x)


t = tf.constant([[1.0, 2.0]])
Reader().reads(t)
Writer().writes(t)
