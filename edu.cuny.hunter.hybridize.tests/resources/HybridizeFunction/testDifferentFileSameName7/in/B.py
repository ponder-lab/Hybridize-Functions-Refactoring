import tensorflow as tf


class Test2:

    @tf.function
    def b(self, a):
        pass


if __name__ == "__main__":
    Test2().b(5)
