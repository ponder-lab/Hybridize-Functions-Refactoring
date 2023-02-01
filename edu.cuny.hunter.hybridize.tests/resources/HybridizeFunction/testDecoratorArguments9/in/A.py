import tensorflow as tf


@tf.function(jit_compile=False)
def func():
  pass


if __name__ == '__main__':
    func()
