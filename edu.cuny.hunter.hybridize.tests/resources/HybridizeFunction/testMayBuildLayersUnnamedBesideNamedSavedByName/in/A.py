import os
import tempfile

import tensorflow as tf


class Pair(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(4)
        self.second = tf.keras.layers.Dense(4, name="dense")

    def call(self, x):
        return self.second(self.first(x))


class Normalized(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.LayerNormalization()
        self.second = tf.keras.layers.LayerNormalization(name="layer_normalization")

    def call(self, x):
        return self.second(self.first(x))


class Apart(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(4)
        self.second = tf.keras.layers.Dense(4, name="proj")

    def call(self, x):
        return self.second(self.first(x))


class Aliased(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Convolution2D(4, 1)
        self.second = tf.keras.layers.Conv2D(4, 1, name="conv2d")

    def call(self, x):
        return tf.reshape(
            self.second(self.first(tf.reshape(x, [-1, 2, 2, 1]))), [-1, 16]
        )


class Counted(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(4)
        self.second = tf.keras.layers.Dense(4)
        self.third = tf.keras.layers.Dense(4, name="dense_1")

    def call(self, x):
        return self.third(self.second(self.first(x)))


class MyDense(tf.keras.layers.Dense):
    pass


class Subclassed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = MyDense(4)
        self.second = tf.keras.layers.Dense(4, name="dense")

    def call(self, x):
        return self.second(self.first(x))


class TemplateUnnamed(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(4)
        self.second = tf.keras.layers.Dense(4)
        self.rest = []
        for i in range(2):
            self.rest.append(tf.keras.layers.Dense(4, name="dense_{}".format(i)))

    def call(self, x):
        x = self.second(self.first(x))
        for dense in self.rest:
            x = dense(x)
        return x


class DenseTemplateNoUnderscore(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = tf.keras.layers.Dense(4)
        self.rest = []
        for i in range(2):
            self.rest.append(tf.keras.layers.Dense(4, name="dense{}".format(i)))

    def call(self, x):
        x = self.first(x)
        for dense in self.rest:
            x = dense(x)
        return x


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.pair = Pair(name="pair")
        self.normalized = Normalized(name="normalized")
        self.apart = Apart(name="apart")
        self.aliased = Aliased(name="aliased")
        self.counted = Counted(name="counted")
        self.subclassed = Subclassed(name="subclassed")
        self.template_unnamed = TemplateUnnamed(name="template_unnamed")
        self.no_underscore = DenseTemplateNoUnderscore(name="no_underscore")

    def call(self, x):
        return (
            self.template_unnamed(
                self.subclassed(self.counted(self.apart(self.normalized(self.pair(x)))))
            ),
            self.aliased(x),
            self.no_underscore(x),
        )


model = Net()
model(tf.ones([2, 4]))
model(tf.ones([3, 4]))
model.save_weights(os.path.join(tempfile.mkdtemp(), "t.h5"))
