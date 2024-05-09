import tensorflow as tf


@tf.function(experimental_follow_type_hints=False)
def func(x: tf.Tensor):
    pass


if __name__ == "__main__":
    func(5)
