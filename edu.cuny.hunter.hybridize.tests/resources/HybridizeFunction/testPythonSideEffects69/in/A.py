import tensorflow as tf


class CachingConv(tf.keras.layers.Layer):
    def __init__(self):
        super(CachingConv, self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight("w", shape=[2, 2])
        self.cached_result = None
        self.built = True

    def call(self, inputs):
        if self.cached_result is None:
            self.cached_result = tf.matmul(inputs, self.w)
        return self.cached_result


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = CachingConv()

    def call(self, inputs):
        return self.layer(inputs)


model = MyModel()
result = model(tf.ones([2, 2]))
