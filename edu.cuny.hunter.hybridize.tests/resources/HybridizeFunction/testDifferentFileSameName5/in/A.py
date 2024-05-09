import tensorflow as tf


class Test:

    @tf.function
    def b(self, a):
        pass


if __name__ == "__main__":
    Test().b(tf.ones([1, 2]))
