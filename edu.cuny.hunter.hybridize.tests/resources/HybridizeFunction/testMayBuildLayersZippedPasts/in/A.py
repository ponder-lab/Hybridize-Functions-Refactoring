import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4)

    def call(self, x, layer_past):
        return self.d(x), x


class Stack(tf.keras.layers.Layer):

    def __init__(self, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        self.blocks = []
        for i in range(self.num_layers):
            self.blocks.append(Block(name="h{}".format(i)))

    def call(self, x, past=None):
        presents = []
        pasts = (
            tf.unstack(past, axis=1) if past is not None else [None] * self.num_layers
        )
        for i, (block, layer_past) in enumerate(zip(self.blocks, pasts)):
            x, present = block(x, layer_past)
            presents.append(present)
        return x, tf.stack(presents, axis=1)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stack = Stack(3, name="stack")

    def call(self, x, past=None):
        return self.stack(x, past)


model = Net()
x = tf.ones([2, 4])
_, presents = model(x)
model(x, presents)
model(tf.ones([3, 4]))
