import tensorflow as tf


@tf.function(input_signature=[])
def func():
    return tf.constant(1)


if __name__ == "__main__":
    func()
