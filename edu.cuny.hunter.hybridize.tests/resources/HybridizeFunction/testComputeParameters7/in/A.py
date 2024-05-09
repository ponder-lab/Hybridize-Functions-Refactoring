import tensorflow as tf


@tf.function(autograph=False)
def func():
    print("Testing")


if __name__ == "__main__":
    func()
