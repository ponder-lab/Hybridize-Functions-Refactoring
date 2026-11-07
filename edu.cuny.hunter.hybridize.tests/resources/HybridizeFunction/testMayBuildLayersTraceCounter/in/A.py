import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = tf.keras.layers.Dense(4)

    def build(self, input_shape):
        self.lazy = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.lazy(self.eager(x))


class Top(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = Block(name="h0")
        self.second = Block(name="h1")

    def call(self, x):
        return self.second(self.first(x))


class LazyBlock(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = tf.keras.layers.Dense(4)
        self.second = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.second(self.first(x))


class LazyTop(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.block = LazyBlock(name="h0")

    def call(self, x):
        return self.block(x)


def make_dense():
    return tf.keras.layers.Dense(4)


class HelperBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = make_dense()

    def build(self, input_shape):
        self.lazy = make_dense()

    def call(self, x):
        return self.lazy(self.eager(x))


class HelperTop(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = HelperBlock(name="h0")
        self.second = HelperBlock(name="h1")

    def call(self, x):
        return self.second(self.first(x))


def wrap1():
    return make_dense()


class WrapBlock1(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = wrap1()

    def build(self, input_shape):
        self.lazy = wrap1()

    def call(self, x):
        return self.lazy(self.eager(x))


class WrapTop1(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = WrapBlock1(name="h0")
        self.second = WrapBlock1(name="h1")

    def call(self, x):
        return self.second(self.first(x))


def wrap2():
    return wrap1()


class WrapBlock2(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = wrap2()

    def build(self, input_shape):
        self.lazy = wrap2()

    def call(self, x):
        return self.lazy(self.eager(x))


class WrapTop2(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = WrapBlock2(name="h0")
        self.second = WrapBlock2(name="h1")

    def call(self, x):
        return self.second(self.first(x))


def wrap3():
    return wrap2()


class WrapBlock3(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = wrap3()

    def build(self, input_shape):
        self.lazy = wrap3()

    def call(self, x):
        return self.lazy(self.eager(x))


class WrapTop3(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = WrapBlock3(name="h0")
        self.second = WrapBlock3(name="h1")

    def call(self, x):
        return self.second(self.first(x))


def wrap4():
    return wrap3()


class WrapBlock4(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = wrap4()

    def build(self, input_shape):
        self.lazy = wrap4()

    def call(self, x):
        return self.lazy(self.eager(x))


class WrapTop4(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = WrapBlock4(name="h0")
        self.second = WrapBlock4(name="h1")

    def call(self, x):
        return self.second(self.first(x))


def wrap5():
    return wrap4()


class WrapBlock5(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = wrap5()

    def build(self, input_shape):
        self.lazy = wrap5()

    def call(self, x):
        return self.lazy(self.eager(x))


class WrapTop5(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = WrapBlock5(name="h0")
        self.second = WrapBlock5(name="h1")

    def call(self, x):
        return self.second(self.first(x))


def wrap6():
    return wrap5()


class WrapBlock6(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = wrap6()

    def build(self, input_shape):
        self.lazy = wrap6()

    def call(self, x):
        return self.lazy(self.eager(x))


class WrapTop6(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = WrapBlock6(name="h0")
        self.second = WrapBlock6(name="h1")

    def call(self, x):
        return self.second(self.first(x))


class MethodBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eager = self.make()

    def build(self, input_shape):
        self.lazy = self.make()

    def make(self):
        return tf.keras.layers.Dense(4)

    def call(self, x):
        return self.lazy(self.eager(x))


class MethodTop(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.first = MethodBlock(name="h0")
        self.second = MethodBlock(name="h1")

    def call(self, x):
        return self.second(self.first(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.top = Top(name="top")
        self.lazy_top = LazyTop(name="lazy_top")
        self.helper_top = HelperTop(name="helper_top")
        self.wrap_top1 = WrapTop1(name="wrap_top1")
        self.wrap_top2 = WrapTop2(name="wrap_top2")
        self.wrap_top3 = WrapTop3(name="wrap_top3")
        self.wrap_top4 = WrapTop4(name="wrap_top4")
        self.wrap_top5 = WrapTop5(name="wrap_top5")
        self.wrap_top6 = WrapTop6(name="wrap_top6")
        self.method_top = MethodTop(name="method_top")

    def call(self, x):
        x = self.helper_top(self.lazy_top(self.top(x)))
        x = self.wrap_top6(
            self.wrap_top5(
                self.wrap_top4(self.wrap_top3(self.wrap_top2(self.wrap_top1(x))))
            )
        )
        return self.method_top(x)


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
