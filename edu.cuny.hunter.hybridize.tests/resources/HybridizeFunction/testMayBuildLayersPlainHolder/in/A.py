import tensorflow as tf


class Head:

    def __init__(self):
        self.d = tf.keras.layers.Dense(2)

    def run(self, x):
        return self.d(x)


class Selected(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.choice = 0
        self.heads = [Head() for _ in range(3)]

    def select(self, i):
        self.choice = i

    def call(self, x):
        return self.heads[self.choice].run(x)


class First(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.heads = [Head() for _ in range(3)]

    def call(self, x):
        return self.heads[0].run(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.selected = Selected()
        self.first = First()

    def call(self, x):
        return self.first(self.selected(x))


model = Net()
model(tf.ones([2, 4]))
model.selected.select(1)
model(tf.ones([3, 4]))
