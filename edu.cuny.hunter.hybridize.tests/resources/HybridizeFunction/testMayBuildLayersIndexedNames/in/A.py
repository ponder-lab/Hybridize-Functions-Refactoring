import tensorflow as tf


class MLP(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.c_fc = tf.keras.layers.Dense(8, name="c_fc")
        self.c_proj = tf.keras.layers.Dense(4, name="c_proj")

    def call(self, x):
        return self.c_proj(self.c_fc(x))


class Attention(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.c_attn = tf.keras.layers.Dense(4, name="c_attn")
        self.c_proj = tf.keras.layers.Dense(4, name="c_proj")

    def call(self, x):
        return self.c_proj(self.c_attn(x))


class Block(tf.keras.layers.Layer):

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.ln_1 = tf.keras.layers.LayerNormalization(name="ln_1")
        self.attn = Attention(name="attn")
        self.mlp = MLP(name="mlp")

    def call(self, x):
        return self.mlp(self.attn(self.ln_1(x)))


class Indexed(tf.keras.layers.Layer):

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        self.blocks = []
        for i in range(self.n):
            self.blocks.append(Block(name="h{}".format(i)))
        self.ln_f = tf.keras.layers.LayerNormalization(name="ln_f")

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


class Constant(tf.keras.layers.Layer):

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        self.blocks = []
        for i in range(self.n):
            self.blocks.append(Block(name="h"))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TwoLoops(tf.keras.layers.Layer):

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        self.first = []
        for i in range(self.n):
            self.first.append(Block(name="h{}".format(i)))
        self.second = []
        for i in range(self.n):
            self.second.append(Block(name="h{}".format(i)))

    def call(self, x):
        for block in self.first:
            x = block(x)
        for block in self.second:
            x = block(x)
        return x


class Formatted(tf.keras.layers.Layer):

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        self.blocks = []
        for i in range(self.n):
            self.blocks.append(Block(name=f"h{i}"))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Nested(tf.keras.layers.Layer):

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        self.blocks = []
        for i in range(self.n):
            for _ in range(2):
                self.blocks.append(Block(name="h{}".format(i)))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Shadowed(tf.keras.layers.Layer):

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        self.blocks = []
        for i in range(self.n):
            self.blocks.append(Block(name="h{}".format(i)))
        self.extra = Block(name="h1")

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return self.extra(x)


class Restarted(tf.keras.layers.Layer):

    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        self.blocks = []
        for _ in range(self.n):
            self.blocks.append(Block(name="h{}".format(next(iter(range(self.n))))))

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.indexed = Indexed(3, name="indexed")
        self.constant = Constant(3, name="constant")
        self.two_loops = TwoLoops(2, name="two_loops")
        self.formatted = Formatted(3, name="formatted")
        self.nested = Nested(2, name="nested")
        self.shadowed = Shadowed(3, name="shadowed")
        self.restarted = Restarted(2, name="restarted")

    def call(self, x):
        x = self.formatted(self.two_loops(self.constant(self.indexed(x))))
        return self.restarted(self.shadowed(self.nested(x)))


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
