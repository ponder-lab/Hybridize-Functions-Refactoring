import tensorflow as tf


class Indexed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.choice = 0
        self.blocks = [tf.keras.layers.Dense(4) for _ in range(3)]

    def select(self, i):
        self.choice = i

    def call(self, x):
        return self.blocks[self.choice](x)


class Single(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.d(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.indexed = Indexed()
        self.single = Single()

    def call(self, x):
        return self.single(self.indexed(x))


model = Net()
model(tf.ones([2, 4]))
model.indexed.select(1)
model(tf.ones([3, 4]))
