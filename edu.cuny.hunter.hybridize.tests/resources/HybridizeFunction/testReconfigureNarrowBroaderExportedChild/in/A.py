# A broader supplied signature on a method of an object the saved object tracks (#808). Saving an object exports the functions of the
# objects it tracks too, such as a submodule, so the narrowing is declined.
import tensorflow as tf


class Encoder(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


class Root(tf.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()


if __name__ == "__main__":
    root = Root()
    root.encoder.f(tf.constant(2.0))
    tf.saved_model.save(root, "exported")
