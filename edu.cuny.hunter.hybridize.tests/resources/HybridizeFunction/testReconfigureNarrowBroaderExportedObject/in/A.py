# A broader supplied signature on a method of an object the program saves as a SavedModel (#808). Saving the object exports each of its
# tf.function attributes, so the method's signature is part of the exported interface and the narrowing is declined.
import tensorflow as tf


class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    tf.saved_model.save(m, "exported")
