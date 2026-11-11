import os
import tempfile

import tensorflow as tf


class Inner(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = tf.keras.layers.Dense(4, name="dense")

    def call(self, x):
        return self.d(x)


class Stack(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first = Inner()
        self.second = Inner()

    def call(self, x):
        return self.second(self.first(x))


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.stack = Stack()

    def call(self, x):
        return self.stack(x)


class Checkpoint(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(os.path.join(tempfile.mkdtemp(), "w"))


model = Net()
model(tf.ones([2, 4]))
model.compile(optimizer="sgd", loss="mse")
model.fit(
    tf.ones([2, 4]), tf.ones([2, 4]), epochs=1, verbose=0, callbacks=[Checkpoint()]
)
