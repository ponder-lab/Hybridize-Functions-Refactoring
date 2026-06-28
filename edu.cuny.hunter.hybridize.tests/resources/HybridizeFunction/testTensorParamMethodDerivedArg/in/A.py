import tensorflow as tf


# An instance method whose arguments are DERIVED tensors (one a method-call result, one a threaded
# parameter). `real` and `pred` should both be tensor-typed.
class M:
    def call(self, x):
        return x * 2.0

    def loss(self, real, pred):
        return tf.reduce_mean(tf.square(pred - real))

    def step(self, inputs, targets):
        pred = self.call(inputs)
        return self.loss(targets, pred)


M().step(tf.constant([1.0, 2.0]), tf.constant([1.0, 1.0]))
