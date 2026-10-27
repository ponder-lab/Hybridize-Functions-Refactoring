import tensorflow as tf


class ProjectMixin:

    def project(self, x):
        return x * 2.0


class Base(tf.keras.Model):
    pass


class Mid(ProjectMixin, Base):
    pass


class Net(Mid):

    def __init__(self):
        super().__init__()
        self.d = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.d(self.project(x))


Net()(tf.ones([2, 4]))
