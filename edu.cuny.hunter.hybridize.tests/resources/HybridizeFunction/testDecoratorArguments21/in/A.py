import tensorflow as tf


@tf.function(experimental_follow_type_hints=False)
def func():
  pass


if __name__ == '__main__':
    func()
