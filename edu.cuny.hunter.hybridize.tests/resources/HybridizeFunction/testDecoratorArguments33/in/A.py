import tensorflow as tf

var = "embedded_matmul"


@tf.function(experimental_implements=var)
def func():
  pass


if __name__ == '__main__':
    func()
