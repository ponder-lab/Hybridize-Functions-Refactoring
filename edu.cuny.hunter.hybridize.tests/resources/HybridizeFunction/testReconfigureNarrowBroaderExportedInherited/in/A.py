# A broader supplied signature on a method a saved object inherits (#808). The method is called through an instance of the base class,
# and an instance of the subclass is saved. Saving it exports the inherited method too, so the narrowing is declined.
import tensorflow as tf


class Base(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def f(self, t):
        return t + 1


class Sub(Base):
    pass


if __name__ == "__main__":
    Base().f(tf.constant(2.0))
    tf.saved_model.save(Sub(), "exported")
