import tensorflow as tf


@tf.function  # This is not the "function" from TensorFlow. Instead, it's a module in the same directory.
def func1():
    pass


if __name__ == "__main__":
    func1()
