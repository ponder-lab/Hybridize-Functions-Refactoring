# A broader supplied signature on a method of a Keras model the program saves through the model's own save method (#808). Saving the
# model exports it. A save call on an object is not always a model saving itself, so the method is possibly exported, and the narrowing
# is declined.
import tensorflow as tf


class M(tf.keras.Model):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    m.save("exported")
