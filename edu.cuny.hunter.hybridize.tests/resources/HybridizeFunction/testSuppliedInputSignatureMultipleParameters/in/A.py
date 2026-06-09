import tensorflow as tf


@tf.function(
    input_signature=[tf.TensorSpec((2, 3), tf.float32), tf.TensorSpec((5,), tf.int32)]
)
def func(a, b):
    return a, b


if __name__ == "__main__":
    func(tf.ones([2, 3]), tf.ones([5], dtype=tf.int32))
