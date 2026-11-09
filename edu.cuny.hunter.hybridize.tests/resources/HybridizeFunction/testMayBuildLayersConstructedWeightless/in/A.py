import tensorflow as tf


class Attention(tf.keras.layers.Layer):

    def call(self, x):
        return tf.nn.softmax(x)


class Scale(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.w = self.add_weight(shape=(), initializer="ones")

    def call(self, x):
        return self.w * x


class Heads(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(4)

    def call(self, x):
        return Attention()(self.proj(x))


class Scaled(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(4)

    def call(self, x):
        return Scale()(self.proj(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.heads = Heads()
        self.scaled = Scaled()

    def call(self, x):
        return self.scaled(self.heads(x))


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
