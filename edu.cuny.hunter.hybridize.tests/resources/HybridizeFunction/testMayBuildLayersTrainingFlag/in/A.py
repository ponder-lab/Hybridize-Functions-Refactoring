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


class ServedGate(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(2)
        self.b = tf.keras.layers.Dense(2)

    def call(self, x, training=None):
        x = self.a(x)

        if training:
            x = self.b(x)

        return x


class Trained(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.gate = TrainedGate()

    def call(self, x, training=None):
        return self.gate(x, training=training)


class Served(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.gate = ServedGate()

    def call(self, x, training=None):
        return self.gate(x, training=training)


xs = tf.ones((4, 3))
ys = tf.ones((4, 2))

trained = Trained()
trained.compile(optimizer="sgd", loss="mse")
trained.fit(xs, ys, epochs=1, verbose=0)

served = Served()
served.compile(optimizer="sgd", loss="mse")
served.predict(xs, verbose=0)
served.fit(xs, ys, epochs=1, verbose=0)
