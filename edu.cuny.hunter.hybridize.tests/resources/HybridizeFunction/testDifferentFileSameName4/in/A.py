import tensorflow as tf


class Test:

    @tf.function
    def b(self):
        pass


if __name__ == "__main__":
    Test().b()
