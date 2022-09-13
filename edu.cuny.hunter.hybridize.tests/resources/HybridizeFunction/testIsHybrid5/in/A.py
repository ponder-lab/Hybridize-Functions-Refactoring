import tensorflow as tf

@tf.function  # This is not the "function" from TensorFlow. Instead, it's a module in the same directory. But, since it has the same signature, there's not much we can do about it.
def func1():
    pass

if __name__ == '__main__':
    func1()
