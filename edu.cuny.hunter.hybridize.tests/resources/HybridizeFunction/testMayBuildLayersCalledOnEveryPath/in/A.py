import tensorflow as tf


class Linear(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = tf.keras.layers.Dense(4)

    def call(self, x, y=None):
        if y is None:
            return self.linear(x)

        return self.linear(x) + y


class Repeated(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(4)

    def call(self, x, flag):
        x = self.a(x)

        if flag:
            x = self.a(x)

        return x


class Checked(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = tf.keras.layers.Dense(4)

    def call(self, x, y):
        if y is None:
            raise ValueError("y must be given.")

        return self.a(x) + y


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.linear = Linear()
        self.repeated = Repeated()
        self.checked = Checked()

    def call(self, x, y, flag):
        return self.checked(self.repeated(self.linear(x, y), flag), x)


model = Net()
model(tf.ones([2, 4]), None, False)
model(tf.ones([2, 4]), tf.ones([2, 4]), True)
