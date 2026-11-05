import tensorflow as tf

USE_A = True


class Chosen(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(4)
        self.b = tf.keras.layers.Dense(4)

    def call(self, x, flag):
        return (self.a if flag else self.b)(x)


class Aliased(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(4)

    def call(self, x):
        f = self.a
        return f(x)


class Configured(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(4)
        self.b = tf.keras.layers.Dense(4)

    def call(self, x):
        return (self.a if USE_A else self.b)(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.chosen = Chosen()
        self.aliased = Aliased()
        self.configured = Configured()

    def call(self, x, flag):
        return self.configured(self.aliased(self.chosen(x, flag)))


model = Net()
model(tf.ones([2, 4]), False)
model(tf.ones([2, 4]), True)
