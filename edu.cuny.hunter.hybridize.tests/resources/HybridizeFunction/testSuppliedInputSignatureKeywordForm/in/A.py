import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(5,), dtype=tf.int32)])
def func(t):
    return t + 1


if __name__ == "__main__":
    func(tf.ones([5], dtype=tf.int32))
