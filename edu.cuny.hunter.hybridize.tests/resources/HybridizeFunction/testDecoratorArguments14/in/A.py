import tensorflow as tf


@tf.function(experimental_implements="embedded_matmul")
def func():
  pass


if __name__ == '__main__':
    func()
