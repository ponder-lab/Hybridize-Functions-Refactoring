# A broader supplied signature on a function the program exports as a SavedModel signature through its concrete function (#808). The
# saved object is a different one, so only the `signatures` argument exports the method, and the narrowing is declined.
import tensorflow as tf


class M(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


class N(tf.Module):
    pass


if __name__ == "__main__":
    m = M()
    m.f(tf.constant(2.0))
    tf.saved_model.save(N(), "exported", signatures=m.f.get_concrete_function())
