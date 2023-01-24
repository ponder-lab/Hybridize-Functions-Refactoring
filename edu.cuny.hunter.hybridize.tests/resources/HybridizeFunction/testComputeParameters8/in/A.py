import tensorflow as tf


@tf.function(experimental_compile=True)
def func():
    print("Testing")


if __name__ == '__main__':
    func()
