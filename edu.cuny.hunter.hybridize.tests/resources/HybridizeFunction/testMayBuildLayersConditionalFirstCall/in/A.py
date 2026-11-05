import tensorflow as tf


class Gated(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(4)
        self.b = tf.keras.layers.Dense(4)

    def call(self, x, flag):
        x = self.a(x)

        if flag:
            x = self.b(x)

        return x


class Toggled(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_b = False
        self.a = tf.keras.layers.Dense(4)
        self.b = tf.keras.layers.Dense(4)

    def enable(self):
        self.use_b = True

    def call(self, x):
        x = self.a(x)

        if self.use_b:
            x = self.b(x)

        return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.gated = Gated()
        self.toggled = Toggled()

    def call(self, x, flag):
        return self.toggled(self.gated(x, flag))


model = Net()
model(tf.ones([2, 4]), False)
model.toggled.enable()
model(tf.ones([3, 4]), True)
