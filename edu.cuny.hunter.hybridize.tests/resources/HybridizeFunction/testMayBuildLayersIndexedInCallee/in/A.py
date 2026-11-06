import tensorflow as tf


class Indexed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.choice = 0
        self.blocks = [tf.keras.layers.Dense(4) for _ in range(3)]

    def select(self, i):
        self.choice = i

    def call(self, x):
        return self._apply(x, self.choice)

    def _apply(self, x, i):
        return self.blocks[i](x)


class Fixed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [tf.keras.layers.Dense(4) for _ in range(3)]

    def call(self, x):
        return self._apply(x)

    def _apply(self, x):
        return self.blocks[0](x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.indexed = Indexed()
        self.fixed = Fixed()

    def call(self, x):
        return self.fixed(self.indexed(x))


model = Net()
model(tf.ones([2, 4]))
model.indexed.select(1)
model(tf.ones([3, 4]))
