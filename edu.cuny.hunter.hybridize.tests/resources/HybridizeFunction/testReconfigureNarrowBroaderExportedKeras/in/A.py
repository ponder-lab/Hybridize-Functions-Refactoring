# A broader supplied signature on a method of a Keras model the program converts to a TensorFlow Lite model (#808). Converting the model
# exports it, so the method's signature is part of the exported interface and the narrowing is declined.
import tensorflow as tf


class M(tf.keras.Model):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    converter = tf.lite.TFLiteConverter.from_keras_model(m)
