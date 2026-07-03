import tensorflow as tf


class CachingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CachingLayer, self).__init__()
        self.cached_result = None

    def call(self, inputs):
        self.cached_result = tf.nn.relu(inputs)
        return self.cached_result


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = CachingLayer()

    def call(self, inputs):
        return self.layer(inputs)


model = MyModel()
result = model(tf.ones([2, 2]))
