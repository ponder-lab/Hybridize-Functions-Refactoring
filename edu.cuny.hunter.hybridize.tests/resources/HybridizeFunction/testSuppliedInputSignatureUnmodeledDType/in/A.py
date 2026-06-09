import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=(2,), dtype=tf.complex64)])
def func(t):
    return t


if __name__ == "__main__":
    func(tf.constant([1 + 2j, 3 + 4j], dtype=tf.complex64))
