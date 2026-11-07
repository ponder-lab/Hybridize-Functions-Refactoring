import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4)

    def project(self, x):
        return self.d(x)


class UnboundVarying(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = Block(name="a")
        self.b = Block(name="b")

    def run(self, blk, x):
        return Block.project(blk, x)


class BoundVarying(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = Block(name="a")
        self.b = Block(name="b")

    def run(self, blk, x):
        return blk.project(x)


class UnboundFixed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = Block(name="a")
        self.b = Block(name="b")

    def run(self, blk, x):
        return Block.project(self.a, x)


class UnboundChosen(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = Block(name="a")
        self.b = Block(name="b")

    def run(self, flag, x):
        return Block.project(self.a if flag else self.b, x)


x = tf.ones([2, 3])
unbound_varying = UnboundVarying()
unbound_varying.run(unbound_varying.a, x)
unbound_varying.run(unbound_varying.b, x)
bound_varying = BoundVarying()
bound_varying.run(bound_varying.a, x)
bound_varying.run(bound_varying.b, x)
unbound_fixed = UnboundFixed()
unbound_fixed.run(unbound_fixed.a, x)
unbound_fixed.run(unbound_fixed.b, x)
unbound_chosen = UnboundChosen()
unbound_chosen.run(True, x)
unbound_chosen.run(False, x)
