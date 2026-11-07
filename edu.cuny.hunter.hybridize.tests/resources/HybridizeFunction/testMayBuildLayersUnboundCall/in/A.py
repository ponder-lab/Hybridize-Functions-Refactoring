import tensorflow as tf


class Shared(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(10, 4)

    def project(self, inputs):
        return self.embedding(inputs)


class Bound(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(10, 4)

    def project(self, inputs):
        return self.embedding(inputs)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.shared_a = Shared()
        self.shared_b = Shared()
        self.bound_a = Bound()
        self.bound_b = Bound()

    def call(self, inputs):
        x = Shared.project(self.shared_a, inputs) + Shared.project(
            self.shared_b, inputs
        )
        return x + self.bound_a.project(inputs) + self.bound_b.project(inputs)


model = Net()
model(tf.constant([[1, 2]]))
model(tf.constant([[1, 2, 3]]))
