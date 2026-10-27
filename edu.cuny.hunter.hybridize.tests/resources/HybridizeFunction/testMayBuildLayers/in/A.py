import os
import tempfile

import tensorflow as tf


class Inner(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4, name="dense")

    def call(self, x):
        return self.d(x)


class Scale(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.w = self.add_weight(name="w", shape=(input_shape[-1],))

    def call(self, x):
        return x * self.w


class Scaled(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.s = Scale(name="scale")

    def call(self, x):
        return self.s(x)


class Flattener(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f = tf.keras.layers.Flatten()

    def call(self, x):
        return self.f(x)


class Stack(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = Inner()
        self.second = Inner()
        self.third = Scaled()
        self.fourth = Scaled()
        self.flat = Flattener()

    def call(self, x):
        return self.flat(self.fourth(self.third(self.second(self.first(x)))))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stack = Stack()

    def call(self, x):
        return self.stack(x)


model = Net()
model(tf.ones([2, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
