import tensorflow as tf


# Same tuple unpack as testTensorParamTupleUnpackKeras, but via a PLAIN method call (self.call)
# rather than the Keras __call__. Isolates the tuple-element propagation from the Keras dispatch.
# `pred` should be tensor-typed.
class M:
    def call(self, x):
        return x * 2.0, None

    def loss(self, real, pred):
        return tf.reduce_mean(tf.square(pred - real))

    def step(self, inputs, targets):
        predictions, _ = self.call(inputs)
        return self.loss(targets, predictions)


M().step(tf.constant([1.0, 2.0]), tf.constant([1.0, 1.0]))
