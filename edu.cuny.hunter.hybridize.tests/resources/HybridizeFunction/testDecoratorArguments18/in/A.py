import tensorflow as tf


@tf.function(experimental_autograph_options=(tf.autograph.experimental.Feature.EQUALITY_OPERATORS, tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS))
def func():
  pass


if __name__ == '__main__':
    func()
