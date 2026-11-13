import tensorflow as tf


class TrainedGate(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(2)
        self.b = tf.keras.layers.Dense(2)

    def call(self, x, training=None):
        x = self.a(x)

        if training:
            x = self.b(x)

        return x


class SwitchedGate(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(2)
        self.b = tf.keras.layers.Dense(2)

    def call(self, x, training=None):
        x = self.a(x)

        if training:
            x = self.b(x)

        return x


xs = tf.ones((4, 3))

trained = tf.keras.Sequential([TrainedGate()])
trained(xs, training=True)
trained(xs, training=True)

switched = tf.keras.Sequential([SwitchedGate()])
switched(xs, training=False)
switched(xs, training=True)
