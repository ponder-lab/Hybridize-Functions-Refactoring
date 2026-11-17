import tensorflow as tf


class Block(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(this, x):
        return this.dense(x)


class Plain(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, x):
        return self.dense(x)


class Shared(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def project(this, x):
        return this.dense(x)


class Tower(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    @staticmethod
    def apply(tower, x):
        return tower.dense(x)


class Mast(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    @staticmethod
    def apply(mast, x):
        return mast.dense(x)


class Kit:

    @staticmethod
    def run(layer, x):
        return layer(x)


first = Block()
second = Block()
first(tf.ones([2, 4]))
second(tf.ones([2, 4]))

plain_first = Plain()
plain_second = Plain()
plain_first(tf.ones([2, 4]))
plain_second(tf.ones([2, 4]))

shared_first = Shared()
shared_second = Shared()
Shared.project(shared_first, tf.ones([2, 4]))
Shared.project(shared_second, tf.ones([2, 4]))

kit = Kit()
kit_first = tf.keras.layers.Dense(3)
kit_second = tf.keras.layers.Dense(3)
kit.run(kit_first, tf.ones([2, 4]))
kit.run(kit_second, tf.ones([2, 4]))

tower_first = Tower()
tower_second = Tower()
Tower.apply(tower_first, tf.ones([2, 4]))
Tower.apply(tower_second, tf.ones([2, 4]))

mast_first = Mast()
mast_second = Mast()
mast_first.apply(mast_first, tf.ones([2, 4]))
mast_first.apply(mast_second, tf.ones([2, 4]))
