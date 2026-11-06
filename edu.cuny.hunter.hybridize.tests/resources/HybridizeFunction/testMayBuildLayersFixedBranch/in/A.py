import tensorflow as tf

USE_PROJECTION = True


class Gated(tf.keras.layers.Layer):

    def __init__(self, use_projection=True, **kwargs):
        super().__init__(**kwargs)
        self.use_projection = use_projection
        self.a = tf.keras.layers.Dense(4)
        self.b = tf.keras.layers.Dense(4)

    def call(self, x):
        x = self.a(x)

        if self.use_projection:
            x = self.b(x)

        return x


class Configured(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(4)
        self.b = tf.keras.layers.Dense(4)

    def call(self, x):
        x = self.a(x)

        if USE_PROJECTION:
            x = self.b(x)

        return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.gated = Gated()
        self.configured = Configured()

    def call(self, x):
        return self.configured(self.gated(x))


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
