import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec([2, 2], tf.float32)])
def func(t):
    return t + 1


if __name__ == "__main__":
    func(tf.ones([2, 2]))
