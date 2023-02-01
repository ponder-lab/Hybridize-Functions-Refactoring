import tensorflow as tf


@tf.function(experimental_implements="google.embedded_matmul")
def func():
  pass


if __name__ == '__main__':
    func()
