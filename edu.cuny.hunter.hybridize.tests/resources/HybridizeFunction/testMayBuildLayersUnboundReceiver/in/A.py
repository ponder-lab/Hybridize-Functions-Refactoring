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


def helper(blk, x):
    return blk.project(x)


def helper_unbound(blk, x):
    return Block.project(blk, x)


class HelperBound(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = Block(name="a")
        self.b = Block(name="b")

    def run(self, blk, x):
        return helper(blk, x)


class HelperUnbound(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = Block(name="a")
        self.b = Block(name="b")

    def run(self, blk, x):
        return helper_unbound(blk, x)


class HelperFixed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = Block(name="a")
        self.b = Block(name="b")

    def run(self, blk, x):
        return helper(self.a, x)


class TwoHolders(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a = Block()

    def run(self, x):
        return Block.project(self.a, x)


class SetUpChosen(tf.keras.layers.Layer):

    def __init__(self, wide=False, **kwargs):
        super().__init__(**kwargs)
        if wide:
            self.a = Block()
        else:
            self.a = Block()

    def run(self, x):
        return Block.project(self.a, x)


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
helper_bound = HelperBound()
helper_bound.run(helper_bound.a, x)
helper_bound.run(helper_bound.b, x)
helper_unbound_holder = HelperUnbound()
helper_unbound_holder.run(helper_unbound_holder.a, x)
helper_unbound_holder.run(helper_unbound_holder.b, x)
helper_fixed = HelperFixed()
helper_fixed.run(helper_fixed.a, x)
helper_fixed.run(helper_fixed.b, x)
first_holder = TwoHolders()
second_holder = TwoHolders()
first_holder.run(x)
second_holder.run(x)
first_holder.run(tf.ones([3, 3]))
second_holder.run(tf.ones([3, 3]))
narrow = SetUpChosen()
wide = SetUpChosen(wide=True)
narrow.run(x)
wide.run(x)
narrow.run(tf.ones([3, 3]))
wide.run(tf.ones([3, 3]))
