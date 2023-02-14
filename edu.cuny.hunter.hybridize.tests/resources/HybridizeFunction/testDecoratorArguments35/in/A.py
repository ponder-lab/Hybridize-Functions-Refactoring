import tensorflow as tf

var = False

@tf.function(experimental_follow_type_hints=var)
def func():
  pass


if __name__ == '__main__':
    func()