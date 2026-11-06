import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.dense(x)


class Stack(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [Block() for _ in range(3)]

    def call(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class Indexed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 3
        self.blocks = [Block() for _ in range(self.n)]

    def call(self, x):
        for i in range(self.n):
            x = self.blocks[i](x)

        return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stack = Stack()
        self.indexed = Indexed()

    def call(self, x):
        return self.indexed(self.stack(x))


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
