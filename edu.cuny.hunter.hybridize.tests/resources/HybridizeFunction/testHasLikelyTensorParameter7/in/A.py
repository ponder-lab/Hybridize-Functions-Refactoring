import tensorflow as tf


@tf.function(experimental_follow_type_hints=True)
def func(x):
    pass


if __name__ == "__main__":
    func(5)
