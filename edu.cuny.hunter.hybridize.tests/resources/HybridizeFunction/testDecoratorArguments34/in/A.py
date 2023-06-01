import tensorflow as tf

var = (tf.autograph.experimental.Feature.EQUALITY_OPERATORS, tf.autograph.experimental.Feature.BUILTIN_FUNCTIONS)


@tf.function(experimental_autograph_options=var)
def func():
  pass


if __name__ == '__main__':
    func()
