import tensorflow as tf

@tf.function(None, (tf.TensorSpec(shape=[None], dtype=tf.float32),))
def test(x):
    return x

if __name__ == '__main__':
    number = tf.constant([1.0, 1.0])
    print(test(number))
