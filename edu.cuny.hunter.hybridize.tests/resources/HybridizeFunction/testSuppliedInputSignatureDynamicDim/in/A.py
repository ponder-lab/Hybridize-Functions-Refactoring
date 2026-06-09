import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=[None, 32], dtype=tf.float32)])
def func(t):
    return t + 1


if __name__ == "__main__":
    func(tf.ones([4, 32]))
