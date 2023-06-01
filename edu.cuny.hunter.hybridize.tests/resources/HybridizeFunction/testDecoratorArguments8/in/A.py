import tensorflow as tf


@tf.function(jit_compile=True)
def func():
  pass


if __name__ == '__main__':
    func()
