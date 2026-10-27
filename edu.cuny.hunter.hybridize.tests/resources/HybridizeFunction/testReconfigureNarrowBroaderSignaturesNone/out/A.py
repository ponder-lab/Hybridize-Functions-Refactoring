# A broader supplied signature on a method of a class the program does not export, while it saves another object with an explicit
# `signatures=None` (#808). None exports nothing, so the closed-world narrowing still applies.
import tensorflow as tf


class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32)])
    def f(self, t):
        return t + 1


class N(tf.Module):
    pass


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    tf.saved_model.save(N(), "exported", signatures=None)
