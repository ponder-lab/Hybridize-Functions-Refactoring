# A broader supplied signature on a function the program assigns to an attribute of an object it saves (#808). Saving the object exports
# the function through that attribute, so the narrowing is declined.
import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
def f(t):
    return t + 1


class Holder(tf.Module):
    pass


if __name__ == "__main__":
    f(tf.constant(2.0))
    holder = Holder()
    holder.serve = f
    tf.saved_model.save(holder, "exported")
