import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)])
def func(t):
    return t + 1


if __name__ == "__main__":
    func(tf.constant(3))
