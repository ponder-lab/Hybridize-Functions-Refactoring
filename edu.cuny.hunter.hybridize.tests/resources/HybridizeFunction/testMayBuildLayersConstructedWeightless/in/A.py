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


class Scale2(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.w = self.add_weight(shape=(), initializer="ones")

    def call(self, x):
        return self.w * x


class StackedDense(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(4)

    def call(self, x):
        return tf.keras.Sequential([tf.keras.layers.Dense(4)])(self.proj(x))


class StackedScale(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(4)

    def call(self, x):
        return tf.keras.Sequential([Scale2()])(self.proj(x))


class StackedWeightless(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(4)

    def call(self, x):
        return tf.keras.Sequential([Attention()])(self.proj(x))


class Softmax(tf.keras.Model):

    def call(self, x):
        return tf.nn.softmax(x)


class ModelInBody(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(4)

    def call(self, x):
        return Softmax()(self.proj(x))


class WrappedDense(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(4)

    def call(self, x):
        return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4))(
            tf.expand_dims(self.proj(x), 1)
        )


class WrappedWeightless(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.proj = tf.keras.layers.Dense(4)

    def call(self, x):
        return tf.keras.layers.TimeDistributed(Attention())(
            tf.expand_dims(self.proj(x), 1)
        )


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.heads = Heads()
        self.scaled = Scaled()
        self.stacked_dense = StackedDense()
        self.stacked_scale = StackedScale()
        self.stacked_weightless = StackedWeightless()
        self.model_in_body = ModelInBody()
        self.wrapped_dense = WrappedDense()
        self.wrapped_weightless = WrappedWeightless()

    def call(self, x):
        y = self.scaled(self.heads(x))
        y = self.model_in_body(
            self.stacked_weightless(self.stacked_scale(self.stacked_dense(y)))
        )
        return self.wrapped_dense(y), self.wrapped_weightless(x)


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
