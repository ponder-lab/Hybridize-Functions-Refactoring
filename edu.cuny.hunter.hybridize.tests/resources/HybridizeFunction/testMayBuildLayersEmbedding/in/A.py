import tensorflow as tf


class EmbLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.e = tf.keras.layers.Embedding(10, 4)

    def call(self, x):
        return self.e(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.emb = EmbLayer()

    def call(self, x):
        return self.emb(x)


model = Net()
model(tf.constant([[1, 2]]))
