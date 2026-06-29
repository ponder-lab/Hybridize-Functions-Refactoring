import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.bfloat16)])
def func(t):
    return t


if __name__ == "__main__":
    func(tf.constant([1.0, 2.0], dtype=tf.bfloat16))
