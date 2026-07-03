import tensorflow as tf


class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight("w", shape=[2, 2])
        self.built = True

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = MyLayer()

    def call(self, inputs):
        return self.layer(inputs)


model = MyModel()
result = model(tf.ones([2, 2]))
