import tensorflow as tf


@tf.function
def test(x):
    return x


if __name__ == '__main__':
    x = tf.constant(1)
    test(x)
    
