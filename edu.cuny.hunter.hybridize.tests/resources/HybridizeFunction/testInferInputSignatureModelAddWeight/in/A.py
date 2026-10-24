import tensorflow as tf


# Receives the kernel a model subclass creates with `add_weight` as `kernel`.
def project(x, kernel):
    return tf.matmul(x, kernel)


class Model(tf.keras.Model):

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=(4, 2))

    def call(self, x):
        return project(x, self.kernel)


model = Model()
model(tf.ones((5, 4)))
