import custom
import tensorflow as tf


@custom.decorator(input_signature=None)
@tf.function(autograph=False)
def func():
    pass


if __name__ == '__main__':
    func()
